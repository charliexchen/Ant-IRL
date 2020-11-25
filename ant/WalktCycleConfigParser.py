#!/usr/bin/python
import yaml
import numpy as np
from ServoIDConfig import MAX_SERVO_COUNT, get_servo_id

from typing import Union


class WalkCycle:
    def __init__(self, path="WalkConfigs/simple_walk_config.yaml"):
        self.path = path
        self.key_frames = []
        self.wait_frames = []
        self._get_walk_cycle_frames()

    def _get_walk_cycle_frames(self):  # install np typing later to fix hthis
        def get_keyframe(raw_key_frame_):
            key_frame_ = np.zeros(8)
            wait_frames_ = 0
            for key in raw_key_frame:
                if key == "frames_to_next_keyframe":
                    wait_frames_ = raw_key_frame[key]
                else:
                    key_frame_[get_servo_id(key)] = raw_key_frame_[key]
            return key_frame_, wait_frames_

        with open(self.path) as file:
            raw_frames = yaml.load(file, Loader=yaml.FullLoader)

        for raw_key_frame in raw_frames.values():
            key_frame, wait_frames = get_keyframe(raw_key_frame)
            self.key_frames.append(key_frame)
            self.wait_frames.append(wait_frames)

    @staticmethod
    def frame_to_command(frame):
        return {servo_id: servo_pos for servo_id, servo_pos in enumerate(frame)}

    def _get_frames(self):
        def get_steps(wait_frames):
            if wait_frames == 0:
                return np.zeros(1)
            else:
                return np.arange(0.0, 1.0, 1.0 / wait_frames)

        key_frame_count = len(self.key_frames)

        def get_next_key_frame_num(key_frame_num):
            return (key_frame_num + 1) % key_frame_count

        assert key_frame_count > 0, "No keyframes found"
        current_key_frame_num = 0
        while True:
            next_key_frame_num = get_next_key_frame_num(current_key_frame_num)
            current_key_frame = self.key_frames[current_key_frame_num]
            next_key_frame = self.key_frames[next_key_frame_num]
            steps = get_steps(self.wait_frames[current_key_frame_num])

            for step in steps:
                yield current_key_frame + step * (next_key_frame - current_key_frame)
            current_key_frame_num = next_key_frame_num

    def get_commands(self):
        frames = self._get_frames()
        while True:
            current_frame = next(frames)
            yield self.frame_to_command(current_frame)

    def get_training_data(self):
        commands = self._get_frames()
        current_frame = next(commands)
        next_frame = next(commands)
        while True:
            yield current_frame, next_frame
            current_frame = next_frame
            next_frame = next(commands)


if __name__ == "__main__":
    wc = WalkCycle("WalkConfigs/simple_walk_config.yaml")
    walk = wc.get_commands()
    for _ in range(20):
        print(next(walk))
