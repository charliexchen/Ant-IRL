import cv2
import functools
import jax
import numpy as np
import optax
import time
from jax import numpy as jnp

from AntController.AntEnvironment import AntIRLEnvironment
from AntController.HaikuActorCritic import HaikuContinuousActorCritic
from AntController.HaikuPredictor import HaikuPredictor
from AntController.JaxUtils import normal_density
from ServoController.SerialServoController import SerialServoController
from ServoController.WalktCycleConfigParser import WalkCycle


def get_phase_vector(length):
    angle = 2 * np.pi * np.arange(length) / length
    return np.dstack((np.cos(angle), np.sin(angle)))[0]


def full_action_to_quarter_action(action):
    return action[:, [0, 4]]


def get_vector_from_angle(angle):
    return np.column_stack((np.cos(angle), np.sin(angle)))


def get_vector_from_phase(phase, max_phase):
    angle = 2 * np.pi * phase / max_phase
    return np.column_stack((np.cos(angle), np.sin(angle)))


def get_full_action(predictor, angle):
    phase = get_vector_from_angle(angle)
    shifted_phase = get_vector_from_angle(angle + np.pi)
    lf_rb_action = predictor.evaluate(phase)
    rf_lb_action = predictor.evaluate(shifted_phase)
    action = np.zeros(8)
    action[[0, 4]] = lf_rb_action
    action[[3, 7]] = rf_lb_action
    action[[1, 5]] = rf_lb_action
    action[[2, 6]] = lf_rb_action
    return action


def get_full_action_from_sample(predictor, angle, standard_deviation):
    phase_vector = get_vector_from_angle(angle)
    shifted_phase_vector = get_vector_from_angle(angle + np.pi)

    lf_rb_action_mean = predictor.evaluate(phase_vector)
    rf_lb_action_mean = predictor.evaluate(shifted_phase_vector)
    lf_rb_action = np.random.normal(lf_rb_action_mean, standard_deviation)
    rf_lb_action = np.random.normal(rf_lb_action_mean, standard_deviation)
    action = np.zeros(8)
    action[[0, 4]] = lf_rb_action
    action[[3, 7]] = rf_lb_action
    action[[1, 5]] = rf_lb_action
    action[[2, 6]] = lf_rb_action

    lf_rb_action_prob = np.prod(normal_density(lf_rb_action, standard_deviation, lf_rb_action_mean))
    rf_lb_action_prob = np.prod(normal_density(rf_lb_action, standard_deviation, rf_lb_action_mean))
    return lf_rb_action.flatten(), lf_rb_action_prob, rf_lb_action.flatten(), rf_lb_action_prob, action


class HaikuCyclicActorCritic(HaikuContinuousActorCritic):
    def __init__(self, params):
        super().__init__(params)
        self.phase = params["init_step"]
        self.phase_steps = params["phase_steps"]
        self.training_queue = {
            "phases": deque(maxlen=params["actor_queue_size"]),
            "rewards": deque(maxlen=params["actor_queue_size"]),
            "action_probability": deque(maxlen=params["actor_queue_size"]),
            "actions": deque(maxlen=params["actor_queue_size"]),
        }
        self.noise_std = params["noise_std"]

    def step_phase(self):
        self.phase = (self.phase + 1) % self.phase_steps

    def get_angle(self):
        return 2 * np.pi * self.phase / self.phase_steps

    def update_training_queue(self, phase, reward, action, probability):
        self.training_queue['phases'].append(phase)
        self.training_queue['rewards'].append(reward)
        self.training_queue['action_probability'].append(probability)
        self.training_queue['actions'].append(action)

    def get_training_batch(self, batch_size):
        def get_batch(arr, ind):
            return np.asarray([arr[i] for i in ind])

        queue_length = len(self.training_queue['phases'])
        if queue_length > batch_size:
            indices = np.random.choice(range(queue_length), batch_size)
        else:
            indices = range(queue_length)
        batch_data = {}
        for key in self.training_queue:
            batch_data[key] = get_batch(self.training_queue[key], indices)
        return batch_data

    def train_actor_and_critic_on_batch(self, batch_size):
        batch_data = self.get_training_batch(batch_size)
        current_state = get_vector_from_phase(batch_data["phases"], self.phase_steps)
        next_state = get_vector_from_phase(batch_data["phases"] + 1, self.phase_steps)
        value = batch_data["rewards"] + self.discount * self.value_critic.evaluate(next_state).flatten()
        advantage = value - self.value_critic.evaluate(current_state).flatten()

        action_means = self.actor.evaluate(current_state)
        current_policy_probabilties = np.prod(normal_density(action_means, self.noise_std, batch_data["actions"]),
                                              axis=1)
        importance_weight = np.minimum(5, np.divide(current_policy_probabilties, batch_data["action_probability"]))

        importance_weighted_advantage = np.multiply(advantage, importance_weight)

        self.actor.params, self.actor.optimizer_state, actor_loss = self._train_continuous_actor(
            self.actor.params, current_state, batch_data["actions"],
            importance_weighted_advantage.reshape((batch_size, 1)),
            self.actor.optimizer_state
        )
        self.actor.generations += 1

        self.value_critic.params, self.value_critic.optimizer_state, critic_loss = self._train_weighted_critic(
            self.value_critic.params, current_state, value, importance_weight, self.value_critic.optimizer_state)
        self.value_critic.generations += 1

        return actor_loss, critic_loss

    def _weighted_critic_loss(
            self, critic_params, states, values, weights
    ):
        predictions = self.value_critic.net_t.apply(critic_params, states).flatten()
        losses = jnp.square(predictions - values)
        weighted_losses = jnp.multiply(losses, weights)
        return jnp.sum(weighted_losses)

    @functools.partial(jax.jit, static_argnums=0)
    def _train_weighted_critic(
            self, critic_params, states, values, weights, optimizer_state
    ):
        loss, gradient = jax.value_and_grad(self._weighted_critic_loss)(critic_params, states, values, weights)
        update, new_optimizer_state = self.value_critic.optimizer.update(gradient, optimizer_state)
        return optax.apply_updates(critic_params, update), new_optimizer_state, loss

    def run_episode_with_actor(self, env, max_len=100):
        """Runs and episode with the actor, storing the SAR triple along with the values and such"""

        self.phase = self.params["init_step"]
        for t in range(max_len):

            phase_action, phase_probility, anti_phase_action, anti_phase_probility, full_action \
                = get_full_action_from_sample(
                self.actor,
                self.get_angle(),
                self.noise_std
            )

            _state_next, reward, terminal, info = env.step(full_action)

            self.update_training_queue(self.phase, reward, phase_action, phase_probility)
            self.update_training_queue((self.phase + int(self.phase_steps / 2)) % self.phase_steps, reward,
                                       anti_phase_action, anti_phase_probility)
            self.step_phase()
            if terminal:
                break
        self.episode_count += 1
        self.random_action_prob *= self.random_action_prob_decay
        info["episode_length"] = t
        return info


class AntCyclicEnvironment(AntIRLEnvironment):
    SPEED_REWARD_SCALE = 100

    def __init__(
            self,
            port="/dev/ttyUSB0",
            speed=0.5,
            window_name="Ant Actor Critic Simplified",
    ):
        super().__init__(port,
                         speed,
                         window_name,
                         sensors_enabled=False)

    def _step_one(self, action):
        self.servo_controller.send(WalkCycle.frame_to_command(action))
        sensor_data = self.servo_controller.get_data_if_ready()
        if sensor_data is None:
            info["frozen"] = True
            if self.sensors_enabled:
                return None, None, True, info
        terminal = False
        info = {}
        position = self.get_normalised_position()
        new_orientation = self.get_orientation_vector()
        state = np.concatenate((action, position, new_orientation))
        reward = self.reward_from_position(position, new_orientation)
        if position[0] > 0.8:
            terminal = True
            info["success"] = True
        elif abs(position[1] - self.INITIAL_POSITION[1]) > 0.25:
            terminal = True
            info["success"] = False
        if cv2.waitKey(1) & 0xFF == ord("q"):
            info["cv2_term"]

        return (state, reward, terminal, info), position

    def step(self, action, down_sampling=4):
        output = None
        for _ in range(down_sampling):
            output, position = self._step_one(action)
        self.previous_position = position
        return output

    def reward_from_position(self, position, orientation):
        delta = (position - self.previous_position) * self.SPEED_REWARD_SCALE
        return np.dot(delta, orientation)


if __name__ == "__main__":
    import yaml
    from collections import deque

    import pickle

    path = "AntController/configs/fixed_cycle_configs/fixed_cycle_acc_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    aac = HaikuCyclicActorCritic(config)
    aac.actor.name = "fixed_cycle_actor"
    aac.value_critic.name = "fixed_cycle_critic"
    input_dataset = WalkCycle(
        "WalkConfigs/nn_training_walk_config.yaml", speed=0.3
    ).get_all_frames()

    phase = 0
    env = AntCyclicEnvironment()
    data = {
        "actor_loss": [],
        "critic_loss": [],
        "success": [],
        "episode_length": []
    }
    try:
        for generation in range(1000):
            env.set_text("Resetting Environment...")
            env.reset()
            env.set_text(f"Running Episode {generation}")
            info = aac.run_episode_with_actor(env)
            actor_loss, critic_loss = aac.train_actor_and_critic_on_batch(64)
            data["actor_loss"].append(actor_loss)
            data["critic_loss"].append(critic_loss)
            if "success" in info:
                data["success"].append(info["success"])
            else:
                data["success"].append(None)
            if "episode_length" in info:
                data["episode_length"].append(info["episode_length"])
            else:
                data["episode_length"].append(None)
            if generation % 5 == 0:
                print("Generation {} took {} frames".format(generation, data["episode_length"][-1]))
                print("saving weights...")
                #aac.save_weights()

    finally:
        print("saving data...")
        file_name = f"AntController/configs/fixed_frame_data.p"
        #pickle.dump(data, open(file_name, "wb"))
    try:
        while True:
            phase += 2 * np.pi / 10
            action = get_full_action(aac.actor, phase)
            arduino_controller.send({key: value for key, value in enumerate(action)})
            data = arduino_controller.get_data_if_ready()

            time.sleep(0.5)
    finally:
        arduino_controller.send_idle_command()
        arduino_controller.close_ports()
        print("Serial Connection Closed.")
