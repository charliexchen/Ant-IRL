from ServoController.WalkToTargetController import WalkToTargetController
from ServoController.WalktCycleConfigParser import WalkCommand
import cv2, pickle
import numpy as np
from ServoController.WalktCycleConfigParser import WalkCycle
import haiku as hk
import jax


class episode_data:
    def __init__(self, episode_num):
        self.episode_num = episode_num
        self.states, self.actions, self.rewards = [], [], []

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


class AntIRLEnvironment(WalkToTargetController):
    INITIAL_POSITION = (0.2, 0.417)
    RESET_FRAMES = 10

    def __init__(self, port="/dev/ttyUSB0", speed=0.3, window_name="Ant Location"):
        super().__init__(port, speed, window_name)
        self.episode_counter = 0

    def reset_environment(self):
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
        state = np.zeros(8)
        position = self.get_normalised_position()
        data = episode_data(self.episode_counter)
        for _ in range(max_len):
            print(state)
            action = predictor(state)
            self.servo_controller.send(WalkCycle.frame_to_command(action))
            new_position = self.get_normalised_position()
            reward = new_position[0] - position[0]
            data.add_step(state, action, reward)
            if position[0] > 0.8:
                reward += 10
                data.add_step(state, action, reward)
                break
            else:
                data.add_step(state, action, reward)
            state = action
            position = new_position
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


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
    commands = wc._get_frames()
    steps = next(commands)
    net_t = hk.transform(net)
    net_t.init(rng, current_position)
    params = pickle.load(open("configs/Ant/params_10473_gen.p", "rb"))
    evaluate = jax.jit(net_t.apply)


    def pred(current_position):
        return np.random.normal(evaluate(params, None, current_position), 0.2)


    env = AntIRLEnvironment()
    env.reset_environment()
    env.run_episode_with_predictor(pred)
    env.reset_environment()
    env.run_episode_with_predictor(pred)
    env.reset_environment()
    env.end_session()
