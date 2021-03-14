import cv2
import haiku as hk
import jax
import numpy as np
import pickle
from collections import deque

from HaikuHelpers.HaikuPredictor import HaikuPredictor
from ServoController.WalkToTargetController import WalkToTargetController
from ServoController.WalktCycleConfigParser import WalkCommand
from ServoController.WalktCycleConfigParser import WalkCycle


class EpisodeData:
    """
    Simple object to store SAR triples
    """

    def __init__(self, episode_num):
        self.episode_num = episode_num
        self.states, self.actions, self.rewards = [], [], []
        self.terminal_reward = None

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


class AntIRLEnvironment(WalkToTargetController):
    """
    Environment which runs the robot with a neural net, and then resets the position using the fixed walk cycle. This
    class also handles the state, action and reward triples, which will allow us to run the RL algorithm of our choice.
    """

    INITIAL_POSITION = (0.1, 0.417)

    RESET_FRAMES = 10
    RESET_TEXT = "Resetting Environment..."
    RUNNING_TEXT = "Running Episode {}"

    def __init__(
        self,
        port="/dev/ttyUSB0",
        speed=0.5,
        window_name="Ant Actor Critic",
        sensors_enabled=True,
    ):
        super().__init__(port, speed, window_name)
        self.episode_counter = 0
        self.sensors_enabled = sensors_enabled
        self.previous_episode_success = None
        self.previous_position = None
        self.previous_robot_configuration = None

    def set_text(self, text):
        self.locator.text = text

    def set_reset_text(self):
        if self.previous_episode_success is None:
            self.set_text(self.RESET_TEXT)
        if self.previous_episode_success:
            self.set_text("Episode Successful, " + self.RESET_TEXT)
        else:
            self.set_text("Episode Failed, " + self.RESET_TEXT)

    def walk_to_reset_position(self):
        """
        Walk the robot to the environments initial position.
        """
        reset_counter = self.RESET_FRAMES
        self.set_target_normalised(self.INITIAL_POSITION)
        while reset_counter > 0:
            self.go_to_target()
            _sensor_data = self.servo_controller.get_data_if_ready()
            if self.command == WalkCommand.IDLE:
                reset_counter -= 1
            else:
                self.set_target_normalised(self.INITIAL_POSITION)
                reset_counter = self.RESET_FRAMES
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def reward_from_position(self, position):
        forward_reward = position[0] - self.previous_position[0]
        side_penalty = (-abs(position[1] - self.INITIAL_POSITION[1])) - (
            -abs(self.previous_position[1] - self.INITIAL_POSITION[1])
        )
        return forward_reward + side_penalty

    def step_one(self, action, delta_action=False):
        if delta_action:
            action = self.previous_robot_configuration + action
            self.previous_robot_configuration = action
        self.servo_controller.send(WalkCycle.frame_to_command(action))
        sensor_data = self.servo_controller.get_data_if_ready()
        terminal = False
        info = {}
        if sensor_data is None:
            # If sensor data is None, then it means that either the robot is frozen or hasn't finished initalizing
            info["frozen"] = True
            if self.sensors_enabled:
                return None, None, True, info
        position = self.get_normalised_position()
        new_orientation = self.get_orientation_vector()
        if self.sensors_enabled:
            state = self.clean_state((action, sensor_data, position, new_orientation))
        else:
            state = np.concatenate((action, position, new_orientation))
        reward = self.reward_from_position(position)
        if position[0] > 0.8:
            reward += 2
            terminal = True
            info["final_reward"] = 2
            info["final_state"] = "success"
            self.previous_episode_success = True
        elif abs(position[1] - self.INITIAL_POSITION[1]) > 0.25:
            reward += -1
            terminal = True
            info["final_reward"] = -1
            info["final_state"] = "fail"
            self.previous_episode_success = False
        self.previous_position = position
        if cv2.waitKey(1) & 0xFF == ord("q"):
            info["cv2_term"]
        return state, reward, terminal, info

    def step(self, action, down_sampling=3):
        output = None
        for _ in range(down_sampling):
            output = self.step_one(action)
        return output

    def reset(self):
        self.prev_position = None
        self.walk_to_reset_position()
        self.randomise_initial_orientation()
        sensor_data = None
        while sensor_data is None:
            robot_state = self.serial_command
            sensor_data = self.servo_controller.get_data_if_ready()
            position = self.get_normalised_position()
            orientation = self.get_orientation_vector()
            self.previous_position = position
        self.previous_robot_configuration = robot_state
        if self.sensors_enabled:
            return self.clean_state((robot_state, sensor_data, position, orientation))
        else:
            return np.concatenate((robot_state, position, orientation))

    def render(self):
        pass

    def run_episode_with_predictor(self, predictor, value_critic=None, max_len=200):
        """
        Runs an episode, and saves data into an episode data object.
        """
        data = EpisodeData(self.episode_counter)
        self.set_reset_text()
        state = self.reset()
        self.set_text(AntIRLEnvironment.RUNNING_TEXT.format(self.episode_counter))
        for _ in range(max_len):
            if not self.servo_controller.ready:
                return
            action = predictor(state)
            state_next, reward, terminal, info = self.step(action)
            data.add_step(state, action, reward)
            if value_critic is not None:
                print(value_critic.evaluate(self.clean_state(state)))
            state = state_next
            if "frozen" in info:
                # If sensor data is None, then it means that either the robot is frozen or hasn't finished initalizing
                print("Robot was Frozen! Abandoning episode.")
                return
            if "final_reward" in info:
                EpisodeData.terminal_reward = info["final_reward"]
            if terminal:
                break
            if "cv2_term" in info:
                break
        return data
        self.episode_counter += 1
        return data

    def save_data(self, data, path="training_data/", name="Fixed_Walk"):
        file_name = path + name + "_" + str(self.episode_counter)
        pickle.dump(data, open(file_name, "wb"))

    def randomise_initial_orientation(self):
        dir = np.random.randint(2)
        if dir == 0:
            command = WalkCommand.LEFT_TURN
            steps = np.random.randint(5)
        else:
            command = WalkCommand.RIGHT_TURN
            steps = np.random.randint(5)
        for _ in range(steps):
            self.serial_command = self.unified_walk_controller.get_next_step(command)
            self.servo_controller.send(
                {id: com for id, com in enumerate(self.serial_command)}
            )
            _position = self.get_normalised_position()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    @staticmethod
    def clean_state(state):
        state = np.concatenate(state)
        return np.delete(np.asarray(state), 18)


class AntIRLEnvironmentMultiStep(AntIRLEnvironment):
    def __init__(
        self,
        port="/dev/ttyUSB0",
        speed=0.5,
        window_name="Ant Location",
        sensors_enabled=True,
        past_steps=1,
        delta_action=False,
    ):
        super().__init__(port, speed, window_name, sensors_enabled)

        self.past_steps = past_steps
        self.previous_state = None
        self.delta_action = delta_action

    def get_multi_step_state(self, state):
        if self.previous_state is None:
            multi_step_state = np.tile(state, self.past_steps + 1)
        else:
            multi_step_state = np.concatenate((self.previous_state, state))[
                len(state) :
            ]
        self.previous_state = multi_step_state
        return multi_step_state

    def step(self, action, down_sampling=3):
        output = None
        for _ in range(down_sampling):
            output = self.step_one(action, self.delta_action)
        state, reward, terminal, info = output
        multi_step_state = self.get_multi_step_state(state)
        return multi_step_state, reward, terminal, info

    def reset(self):
        state = super().reset()
        self.previous_state = None
        return self.get_multi_step_state(state)


class AntIRLEnvironmentSimplified(AntIRLEnvironment):
    INT_TO_COMMAND_MAP = {
        0: WalkCommand.FORWARD,
        1: WalkCommand.LEFT_TURN,
        3: WalkCommand.RIGHT_TURN,
    }

    def __init__(
        self,
        port="/dev/ttyUSB0",
        speed=0.5,
        window_name="Ant Location",
        sensors_enabled=True,
    ):
        super().__init__(port, speed, window_name, sensors_enabled)
        self.current_action = None

    def step(self, command, down_sampling=10):
        output = None
        walk_command = self.INT_TO_COMMAND_MAP[command]
        for _ in range(down_sampling):
            action = self.unified_walk_controller.get_next_step(walk_command)
            output = self.step_one(action, self.delta_action)
        state, reward, terminal, info = output
        multi_step_state = self.get_multi_step_state(state)
        return multi_step_state, reward, terminal, info

    def reset(self):
        self.prev_position = None
        self.walk_to_reset_position()
        self.randomise_initial_orientation()
        sensor_data = None
        while sensor_data is None:
            sensor_data = self.servo_controller.get_data_if_ready()
            position = self.get_normalised_position()
            orientation = self.get_orientation_vector()
            self.previous_position = position
        return np.concatenate((position, orientation))

    def step_one(self, action):
        self.servo_controller.send(WalkCycle.frame_to_command(action))
        sensor_data = self.servo_controller.get_data_if_ready()
        terminal = False
        info = {}
        if sensor_data is None:
            # If sensor data is None, then it means that either the robot is frozen or hasn't finished initalizing
            info["frozen"] = True
            if self.sensors_enabled:
                return None, None, True, info
        position = self.get_normalised_position()
        new_orientation = self.get_orientation_vector()
        state = self.clean_state((position, new_orientation))
        reward = self.reward_from_position(position)
        if position[0] > 0.8:
            reward += 2
            terminal = True
            info["final_reward"] = 2
            info["final_state"] = "success"
            self.previous_episode_success = True
        elif abs(position[1] - self.INITIAL_POSITION[1]) > 0.25:
            reward += -1
            terminal = True
            info["final_reward"] = -1
            info["final_state"] = "fail"
            self.previous_episode_success = False
        self.previous_position = position
        if cv2.waitKey(1) & 0xFF == ord("q"):
            info["cv2_term"]
        return state, reward, terminal, info


if __name__ == "__main__":
    wc = WalkCycle("walk_cycle_configs/nn_training_walk_config.yaml", speed=0.3)
    frames = wc.get_frames()

    def pred_fixed(_current_position):
        return next(frames)

    critic = HaikuPredictor.get_model_from_saved_file(
        "training_configs/selected_critic.p"
    )
    env = AntIRLEnvironment()
    env.episode_counter = 394
    try:
        for i in range(10000):
            data = env.run_episode_with_predictor(pred_fixed)
            if data is not None:
                env.save_data(data, path="training_data/", name="Fixed_Walk_With_Sensor")
                print(f"Episode {env.episode_counter} saved")
    finally:
        env.walk_to_reset_position()
        env.end_session()
