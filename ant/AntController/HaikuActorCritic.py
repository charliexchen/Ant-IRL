import copy
import functools
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pickle
import time
from collections import deque
from functools import partial
from jax import jit

from AntController.AntEnvironment import EpisodeData
from AntController.HaikuPredictor import HaikuPredictor
from AntController.JaxUtils import normal_density


class EpisodeWithValue(EpisodeData):
    def __init__(self, episode_num, discount, action_space_size):
        super().__init__(episode_num)
        self.discount = discount
        self.values, self.advantages = [], []
        self.action_space_size = action_space_size

    def add_step(self, state, action, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

    def get_discounted_rewards(self):
        labels = []
        discounted_reward = np.asarray([0])
        for reward in self.rewards:
            discounted_reward = discounted_reward * self.discount + reward
            labels.append(discounted_reward)
        return np.asarray(labels[::-1])

    def get_td_label(self):
        """
        Temporal difference labels used to train the critic and to calculate advantage. Seems like np.pad is really slow
        so there is room for improvement here.
        """

        return (
            np.asarray(self.rewards)
            + np.pad(self.values, ((0, 1), (0, 0)), mode="constant")[1:] * self.discount
        )

    def get_advantage(self):
        """
        Advantage is the action-value subtracted by the value. This can be estimated with r_t + yV(s_t+1) - V(s_t)
        """
        self.advantages = self.get_td_label() - np.asarray(self.values)
        return self.advantages

    def get_per_action_advantage(self):
        """
        Maps the vector of actions and advantages to a matrix where the ith row is zero, but with the ith advantage on
        the jth col, where j is the action taken at step i.
        """
        self.get_advantage()
        per_action_advantage = np.zeros((len(self.advantages), self.action_space_size))
        per_action_advantage[
            np.arange(len(self.advantages)), self.actions
        ] = self.advantages.flatten()
        return per_action_advantage

    def get_value_from_predictor(self, predictor):
        self.values = predictor.evaluate(np.asarray(self.states))
        return self.values

    @classmethod
    def create_with_data(cls, episode_num, discount, states, rewards, actions):
        instance = cls(episode_num, discount)
        instance.states, instance.rewards, instance.actions = states, rewards, actions
        return instance


class HaikuActorCritic:
    def __init__(self, params):
        """
        Implementation of Advantage Actor Critic in Jax and Haiku.

        Takes in a config which may be specified in YAML, and then creates the actor critic pair which can be used
        a environment. This class also handles the data which is being generated by the environments so we can have
        consistent batch sizes for the model training.
        """
        self.params = copy.deepcopy(params)
        self.actor = HaikuPredictor.generate_controller_from_config(
            params["actor_config"]
        )
        self.value_critic = HaikuPredictor.generate_controller_from_config(
            params["critic_config"]
        )
        self.episode_count = 0
        self.discount = params["discount"]

        self.actor_training_queue = {
            "states": deque(maxlen=params["actor_queue_size"]),
            "per_action_advantage": deque(maxlen=params["actor_queue_size"]),
        }
        self.critic_training_queue = {
            "states": deque(maxlen=params["critic_queue_size"]),
            "target_values": deque(maxlen=params["critic_queue_size"]),
        }
        self.random_action_prob = params["random_action_prob"]
        if "random_action_prob_half_life" in params:
            self.random_action_prob_decay = 0.5 ** (
                1 / params["random_action_prob_half_life"]
            )
        else:
            self.random_action_prob_decay
        self.action_space_size = params["action_space_size"]

    def train_critic(self):
        if len(self.critic_training_queue["states"]) < self.params["critic_queue_size"]:
            return
        states = np.asarray(self.critic_training_queue["states"])
        target_values = np.asarray(self.critic_training_queue["target_values"])
        return self.value_critic.train_batch(states, target_values)

    @functools.partial(jax.jit, static_argnums=0)
    def _train_actor(self, actor_params, states, per_action_advantage, optimizer_state):
        gradient = jax.grad(self._advantage_scaled_log_likelihoods)(
            actor_params, states, per_action_advantage
        )
        update, new_optimizer_state = self.actor.optimizer.update(
            gradient, optimizer_state
        )
        return optax.apply_updates(actor_params, update), new_optimizer_state

    def _advantage_scaled_log_likelihoods(
        self, actor_params, states, per_action_advantage
    ):
        log_likelihoods = jnp.log(self.actor.net_t.apply(actor_params, states))
        # negative, since we want to gradient ascend the log likelihood * advantage, and the optimizer is descending
        return -jnp.sum(jnp.multiply(per_action_advantage, log_likelihoods))

    def train_actor(self):
        if len(self.actor_training_queue["states"]) < self.params["actor_queue_size"]:
            return
        states = np.asarray(self.actor_training_queue["states"])
        per_action_advantage = np.asarray(
            self.actor_training_queue["per_action_advantage"]
        )
        self.actor.params, self.actor.optimizer_state = self._train_actor(
            self.actor.params, states, per_action_advantage, self.actor.optimizer_state
        )

    def get_sampled_action(self, state, env):
        if np.random.random_sample() < self.random_action_prob:
            return self.get_random_action(env)
        action_distribution = np.array(self.actor.evaluate(state))
        action_distribution[0] = (
            action_distribution[0] - sum(action_distribution) + 1
        )  # This is a bug in numpy :(
        return np.random.choice(env.action_space.n, p=action_distribution)

    def get_optimal_action(self, state):
        action_distribution = np.array(self.actor.evaluate(state))
        return np.argmax(action_distribution)

    def run_episode_with_actor(self, env, render, sampled, max_len=100):
        """Runs and episode with the actor, storing the SAR triple along with the values and such"""
        episode_data = EpisodeWithValue(
            self.episode_count, self.discount, env.action_space.n
        )
        state = env.reset()
        for _t in range(max_len):
            if render:
                env.render()
            if sampled:
                action = self.get_sampled_action(state, env)
            else:
                action = self.get_optimal_action(state)
            state_next, reward, terminal, _info = env.step(action)
            episode_data.add_step(
                state, action, [reward], self.value_critic.evaluate(state)
            )
            state = state_next
            if terminal:
                break
        self.episode_count += 1
        self.random_action_prob *= self.random_action_prob_decay
        return episode_data

    def add_episode_data_to_queue(self, episode_data):
        """
        Appends the training data from an episode to the training queue. This ensures that JAX has a consistent training
        set size.
        """
        value_labels = episode_data.get_td_label()
        self.critic_training_queue["states"].extend(episode_data.states)
        self.critic_training_queue["target_values"].extend(value_labels)

        per_action_advantage = episode_data.get_per_action_advantage()
        self.actor_training_queue["states"].extend(episode_data.states)
        self.actor_training_queue["per_action_advantage"].extend(per_action_advantage)

    def run_episode_and_train(self, env, render, sampled):
        episode_data = self.run_episode_with_actor(env, render, sampled)
        self.add_episode_data_to_queue(episode_data)
        critic_loss = self.train_critic()
        self.train_actor()
        return episode_data, critic_loss

    def get_random_action(self, env):
        return np.random.choice(env.action_space.n)


class HaikuContinuousActorCritic(HaikuActorCritic):
    """
    For AAC in continuous action spaces. This considers the policy to be a normal distribution with the NN output being
    the mean (and possibly the variance in the future)
    """

    def __init__(self, params):
        super().__init__(params)
        self.actor_training_queue = {
            "states": deque(maxlen=params["actor_queue_size"]),
            "advantages": deque(maxlen=params["actor_queue_size"]),
            "actions": deque(maxlen=params["actor_queue_size"]),
        }
        self.noise_std = params["noise_std"]

    def _advantage_scaled_log_likelihoods_normal(
        self, actor_params, states, actions, advantages
    ):
        """
        Calculate the log likelihoods using the PDF of normal dist, then scale with advantage to get the 'loss'.
        """
        means = self.actor._evaluate(actor_params, states)
        log_likelihoods = jnp.log(normal_density(means, self.noise_std, actions))
        advantage_scaled_likelihood = jnp.multiply(log_likelihoods, advantages)
        # negative, since we want to gradient ascend the log likelihood * advantage and the optimizer is descending
        return -jnp.sum(advantage_scaled_likelihood)

    @functools.partial(jax.jit, static_argnums=0)
    def _train_continuous_actor(
        self, actor_params, states, actions, advantages, optimizer_state
    ):
        gradient = jax.grad(self._advantage_scaled_log_likelihoods_normal)(
            actor_params, states, actions, advantages
        )
        update, new_optimizer_state = self.actor.optimizer.update(
            gradient, optimizer_state
        )
        return optax.apply_updates(actor_params, update), new_optimizer_state

    def train_actor(self):
        if len(self.actor_training_queue["states"]) < self.params["actor_queue_size"]:
            return
        states = np.asarray(self.actor_training_queue["states"])
        advantages = np.asarray(self.actor_training_queue["advantages"])
        actions = np.asarray(self.actor_training_queue["actions"])
        self.actor.params, self.actor.optimizer_state = self._train_continuous_actor(
            self.actor.params, states, actions, advantages, self.actor.optimizer_state
        )

    def add_episode_data_to_queue(self, episode_data):
        """
        Appends the training data from an episode to the training queue. This ensures that JAX has a consistent training
        set size.
        """
        value_labels = episode_data.get_td_label()
        self.critic_training_queue["states"].extend(episode_data.states)
        self.critic_training_queue["target_values"].extend(value_labels)

        self.actor_training_queue["states"].extend(episode_data.states)
        self.actor_training_queue["advantages"].extend(episode_data.get_advantage())
        self.actor_training_queue["actions"].extend(episode_data.actions)

    def run_episode_with_actor(self, env, render, sampled, max_len=120):
        """Runs and episode with the actor, storing the SAR triple along with the values"""
        episode_data = EpisodeWithValue(
            self.episode_count, self.discount, self.action_space_size
        )
        state = env.reset()
        for t in range(max_len):
            if render:
                env.render()
            if sampled:
                action = self.get_sampled_action(state)
            else:
                action = self.get_optimal_action(state)
            state_next, reward, terminal, info = env.step(action)
            if t == 99:
                reward += -1
                info["final_state"] = "fail"
            if "frozen" in info:
                episode_data = EpisodeWithValue(
                    self.episode_count, self.discount, self.action_space_size
                )
                env.reset()
                time.sleep(1)
            episode_data.add_step(
                state, action, [reward], self.value_critic.evaluate(state)
            )
            state = state_next
            if terminal:
                break
        self.episode_count += 1
        self.noise_std *= self.random_action_prob_decay
        return episode_data

    def get_sampled_action(self, state):
        mean = self.actor.evaluate(state)
        return np.random.normal(mean, self.noise_std)

    def get_optimal_action(self, state):
        return self.actor.evaluate(state)


if __name__ == "__main__":
    """
    Imports a sample config, and then train actor critic on CartPole. Considers it 'solved' if the moving 
    average over 100 episodes goes over 195, which isn't perfect, but does demonstrate that it's working.

    Generally takes 1.5k episodes to solve.
    """
    import yaml

    np.random.seed(0)
    path = "AntController/configs/cartpole_aac_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    env_ = gym.make("CartPole-v1")
    aac = HaikuActorCritic(config)
    show = False
    rewards = deque(maxlen=100)
    while True:
        episode_data_, value_loss = aac.run_episode_and_train(env_, show, not show)
        print(
            "Episode {} with rewards {}".format(
                aac.episode_count, np.sum(episode_data_.rewards)
            )
        )
        rewards.append(np.sum(episode_data_.rewards))
        if np.mean(rewards) > 195.0 and not show:
            print("Solved!")
            show = True
