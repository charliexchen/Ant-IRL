import copy
import numpy as np
import pickle
import time
import yaml
from collections import deque

from AntController.AntEnvironment import AntIRLEnvironmentMultiStep, AntIRLEnvironment
from HaikuHelpers.HaikuActorCritic import HaikuContinuousActorCritic, EpisodeWithValue
from HaikuHelpers.HaikuPredictor import HaikuPredictor


class AntActorCritic(HaikuContinuousActorCritic):
    """ fixed_cycle_configs with some special logic for the ant environment e.g. rendering text and value outputs, frozen robot
    recovery """

    def __init__(self, params, ant_env):
        super().__init__(params)
        self.ant_env = ant_env
        self.param_queue = {
            "critic_params": deque(maxlen=16),
            "actor_params": deque(maxlen=16),
        }

    def run_episode_with_actor(self, sampled, max_len=120):
        """Runs and episode with the actor, storing the SAR triple along with the values"""
        episode_data = EpisodeWithValue(
            self.episode_count, self.discount, self.action_space_size
        )
        self.ant_env.set_reset_text()
        state = self.ant_env.reset()

        for t in range(max_len):
            if sampled:
                action = self.get_sampled_action(state)
            else:
                action = self.get_optimal_action(state)
            action = np.clip(action, -1.0, 1.0)
            state_next, reward, terminal, info = self.ant_env.step(action)
            if t == max_len - 1:
                reward += -1
                info["final_state"] = "fail"
                self.ant_env.previous_episode_success = False
            if "frozen" in info:
                episode_data = EpisodeWithValue(
                    self.episode_count, self.discount, self.action_space_size
                )
                env.reset()
                time.sleep(1)
            value = self.value_critic.evaluate(state)
            episode_data.add_step(state, action, [reward], value)
            state = state_next
            self.ant_env.locator.text = AntIRLEnvironment.RUNNING_TEXT.format(
                self.episode_count
            ) + ", Value: {:.5f}".format(value[0])

            if terminal:
                break
        self.episode_count += 1
        self.noise_std *= self.random_action_prob_decay
        return episode_data

    def run_episode_and_train(self, sampled):
        episode_data = self.run_episode_with_actor(sampled)
        self.save_params_to_queue()
        self.add_episode_data_to_queue(episode_data)
        critic_loss = self.train_critic()
        self.train_actor()
        return episode_data, critic_loss

    def save_params_to_queue(self):
        self.param_queue["critic_params"].append(
            copy.deepcopy(self.value_critic.params)
        )
        self.param_queue["actor_params"].append(copy.deepcopy(self.actor.params))

    def reset_to_previous_version(self, bad_version_count):
        print("Rolling back to a previous version...")
        for _ in range(bad_version_count):
            self.param_queue["critic_params"].pop()
            self.param_queue["actor_params"].pop()

        self.actor.params = copy.deepcopy(self.param_queue["actor_params"][-1])
        self.value_critic.params = copy.deepcopy(self.param_queue["critic_params"][-1])
        for key in self.actor_training_queue:
            self.actor_training_queue[key].clear()
        for key in self.critic_training_queue:
            self.critic_training_queue[key].clear()
        self.episode_count -= bad_version_count


if __name__ == "__main__":
    path = "Environment/training_configs/fixed_cycle_acc_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    rewards_data, frames_data, critic_losses = [], [], []
    env = AntIRLEnvironment(sensors_enabled=False)
    aac = AntActorCritic(config, env)
    aac.actor.name = "actor_no_sensor"
    aac.value_critic.name = "critic_no_sensor"
    try:
        for episode in range(10000):
            try:
                episode_data, critic_loss = aac.run_episode_and_train(True)

                total_rewards = np.sum(episode_data.rewards)
                frames = len(episode_data.states)
                rewards_data.append(total_rewards)
                frames_data.append(frames)
                critic_losses.append(critic_loss)

                if episode % 5 == 0:
                    aac.save_weights()

            except ValueError:
                print(
                    "Actor has reached a bad state, likely due to exploding gradients."
                )
                aac.reset_to_previous_version(3)
    finally:
        pickle.dump(
            (rewards_data, frames_data, critic_losses),
            open("../training_configs/latest_data.p", "wb"),
        )
