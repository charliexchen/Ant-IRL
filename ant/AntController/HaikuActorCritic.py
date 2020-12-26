import os
import pickle
from functools import partial
import copy
import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import gym
import numpy as np
from HaikuPredictor import HaikuPredictor
from AntController.AntEnvironment import EpisodeData


class EpisodeWithValue(EpisodeData):
    def __init__(self, episode_num, value_predictor, discount):
        super().__init__(episode_num)
        self.discount = discount
        self.predictor = value_predictor
        self.value = value_predictor.evaluate(self.states)
        self.advantage = None, None

    def get_critic_training_labels(self):
        return self.rewards[:-1] + self.value[1:] * self.discount

    def get_advantage(self):
        return self.get_critic_training_labels() - self.value[:-1]


class HaikuActorCritic:
    def __init__(self, params):
        self.params = copy.deepcopy(params)
        self.actor = HaikuPredictor.generate_controller_from_config(params['actor_config'])
        self.value_critic = HaikuPredictor.generate_controller_from_config(params['critic_config'])

    def train_critic(self, episode):
        labels = episode.get_critic_training_labels
        self.value_critic.train_batch(states, labels)

    @jit
    def train_actor(self):
        learning_rate = -self.learning_rate / len(states)
        update = self._jit_grad(self.params, states, labels, learning_rate)
        self.params = jax.tree_multimap(self._jit_sgd, self.params, update)
        self.learning_rate = self.learning_rate * self.decay
        self.generations += 1


if __name__ == "__main__":
    import yaml
    path = "AntController/configs/cartpole_aac_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    aac = HaikuActorCritic(config)

    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    rewards = []
    states = []
    actions = []
    while True:
        state = env.reset()
        while True:
            env.render()
            action_distribution = np.array(aac.actor.evaluate(state))
            action_distribution[0] = action_distribution[0] - sum(action_distribution) + 1  # bug in numpy :(
            action = np.random.choice(action_space, p=action_distribution)
            state = np.reshape(state, [1, observation_space])
            state_next, reward, terminal, info = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = state_next
            if terminal:
                break
        break
    print(len(states), len(rewards), len(actions))