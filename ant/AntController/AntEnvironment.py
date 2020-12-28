import cv2
import haiku as hk
import jax
import numpy as np
import pickle
from collections import deque

from HaikuPredictor import HaikuPredictor
from ServoController.WalkToTargetController import WalkToTargetController
from ServoController.WalktCycleConfigParser import WalkCommand
from ServoController.WalktCycleConfigParser import WalkCycle


class EpisodeData:
    def __init__(self, episode_num):
        self.episode_num = episode_num
        self.states, self.actions, self.rewards = [], [], []

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


class AntIRLEnvironment(WalkToTargetController):
    """
    Environment which runs the robot with a neural net, and then resets the position using the fixed walk cycle. This
    class also handles the state, action and reward triples, which will allow us to run the RL algorith of our choice.
    """
    INITIAL_POSITION = (0.2, 0.417)
    RESET_FRAMES = 10

    def __init__(self, port="/dev/ttyUSB0", speed=0.5, window_name="Ant Location"):
        super().__init__(port, speed, window_name)
        self.episode_counter = 0

    def reset_environment(self):
        """
        Walk the robot to the environments initial position.
        """
        reset_counter = self.RESET_FRAMES
        self.set_target_normalised(self.INITIAL_POSITION)
        while reset_counter > 0:
            self.go_to_target()
            if self.command == WalkCommand.IDLE:
                reset_counter -= 1
            else:
                self.set_target_normalised(self.INITIAL_POSITION)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run_episode_with_predictor(self, predictor, max_len=200):
        """
        Runs an episode, and saves data into an episode data object.
        """
        robot_state = np.zeros(8)
        position = self.get_normalised_position()
        data = EpisodeData(self.episode_counter)
        for _ in range(max_len):
            action = predictor(robot_state)
            self.servo_controller.send(WalkCycle.frame_to_command(action))
            sensor_data = self.servo_controller.get_data()
            new_position = self.get_normalised_position()
            new_orientation = self.get_orientation_vector()
            state = (robot_state, sensor_data, new_position, new_orientation)
            reward = new_position[0] - position[0]
            if position[0] > 0.8:
                reward += 10
                data.add_step(state, action, reward)
                break
            elif position[1] > 0.7 or position[1] < 0.1:
                reward -= 5
                data.add_step(state, action, reward)
                break
            else:
                data.add_step(state, action, reward)
            robot_state = action
            position = new_position
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.save_data(data)
        self.episode_counter += 1

    def save_data(self, data, path='TrainingData/', name="Fixed_Walk"):
        file_name = path + name + "_" + str(self.episode_counter)
        pickle.dump(data, open(file_name, "wb"))


class AntResetDetection:
    """The accelerometer with DMP occasionally glitches and freezes the robot. We need to detect if this happens and
    reset the robot. This way we can leave the robot training for longer. """

    def __init__(self, params, controller):
        self.controller = controller
        self.params = params
        self.position_list = deque(maxlen=params["max_len"])

    def reset_if_stuck(self, pos):
        self.position_list.append(pos)

        self.controller.reset()


if __name__ == "__main__":
    def net(current_position):
        mlp = hk.Sequential([
            hk.Linear(1024), jax.nn.sigmoid,
            hk.Linear(8)
        ])
        return mlp(current_position)


    rng = jax.random.PRNGKey(42)

    wc = WalkCycle()
    input_dataset = wc.get_training_data(1)
    current_position, label = next(input_dataset)
    commands = wc.get_frames()
    steps = next(commands)
    net_t = hk.transform(net)
    net_t.init(rng, current_position)
    params = pickle.load(open("configs/Ant/params_10473_gen.p", "rb"))
    evaluate = jax.jit(net_t.apply)


    def pred(current_position):
        return evaluate(params, None, current_position)


    frames = wc.get_frames()


    def pred_fixed(_current_position):
        return next(frames)


    env = AntIRLEnvironment()
    try:
        for _ in range(100):
            env.reset_environment()
            env.run_episode_with_predictor(pred_fixed)
    finally:
        env.reset_environment()
        env.end_session()
