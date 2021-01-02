#!/usr/bin/python
from enum import Enum
from typing import Optional

import numpy as np
import yaml

from ServoController.ServoIDConfig import MAX_SERVO_COUNT, is_speed_dependent
from ServoController.ServoIDConfig import get_servo_id


class WalkCycle:
    """
    Imports a yaml file which contains the keyframes of a walkcycle, and the handles the interpolation of the frames so
    we can have manually engineer walk cycles.
    """

    def __init__(self, path, speed=1.0):
        self.path = path
        self.key_frames = []
        self.wait_frames = []
        self.speed = speed
        self._get_walk_cycle_frames()
        self.all_frames = None

    def _get_walk_cycle_frames(self):  # install np typing later to fix this typing
        def get_keyframe(raw_key_frame_, speed):
            key_frame_ = np.zeros(MAX_SERVO_COUNT)
            wait_frames_ = 0
            for key in raw_key_frame:
                if key == "frames_to_next_keyframe":
                    wait_frames_ = raw_key_frame[key]
                else:
                    servo_id = get_servo_id(key)
                    multiplier = speed if is_speed_dependent(servo_id) else 1.0
                    key_frame_[servo_id] = raw_key_frame_[key] * multiplier
            return key_frame_, wait_frames_

        with open(self.path) as file:
            raw_frames = yaml.load(file, Loader=yaml.FullLoader)

        for raw_key_frame in raw_frames.values():
            key_frame, wait_frames = get_keyframe(raw_key_frame, self.speed)
            self.key_frames.append(key_frame)
            self.wait_frames.append(wait_frames)

    @staticmethod
    def frame_to_command(frame):
        """
        The frames are numpy arrays since that makes the interpolation easier. Converts the frame into a command which
        SerialServoController can send.
        :param frame:
        :return: dict of servo id to commands
        """
        return {servo_id: servo_pos for servo_id, servo_pos in enumerate(frame)}

    @staticmethod
    def _get_steps(wait_frames):
        if wait_frames == 0:
            return np.zeros(1)
        else:
            return np.arange(0.0, 1.0, 1.0 / wait_frames)

    def get_frames(self):

        key_frame_count = len(self.key_frames)

        def get_next_key_frame_num(key_frame_num):
            return (key_frame_num + 1) % key_frame_count

        assert key_frame_count > 0, "No keyframes found"
        current_key_frame_num = 0
        while True:
            next_key_frame_num = get_next_key_frame_num(current_key_frame_num)
            current_key_frame = self.key_frames[current_key_frame_num]
            next_key_frame = self.key_frames[next_key_frame_num]
            steps = self._get_steps(self.wait_frames[current_key_frame_num])

            for step in steps:
                yield current_key_frame + step * (next_key_frame - current_key_frame)
            current_key_frame_num = next_key_frame_num

    def get_commands(self):
        frames = self.get_frames()
        while True:
            current_frame = next(frames)
            yield self.frame_to_command(current_frame)

    def get_training_data(self, steps=3):
        commands = self.get_frames()
        input_frames = []
        for _ in range(steps):
            input_frames.append(next(commands))
        label_frame = next(commands)
        while True:
            yield np.concatenate(input_frames), label_frame
            input_frames.append(label_frame)
            input_frames.pop(0)
            label_frame = next(commands)

    def _get_all_frames(self):
        frames = []
        frame_generator = self.get_frames()
        for _ in range(sum(self.wait_frames)):
            frames.append(next(frame_generator))
        self.all_frames = frames

    def get_closest_frame_id(self, current_frame):
        self._get_all_frames()
        min_ind = 0
        min_distance = np.linalg.norm(current_frame - self.all_frames[0])
        for i, frame in enumerate(self.all_frames):
            dist = np.linalg.norm(current_frame - self.all_frames[0])
            if dist < min_distance:
                min_ind = i
                min_distance = dist
        return min_ind

    def start_from_position(self, current_frame):
        closest_frame_id = self.get_closest_frame_id(current_frame)
        closest_frame = self.all_frames[closest_frame_id]
        steps = int(max(abs(closest_frame - current_frame)) / 0.4)

        for step in self._get_steps(steps):
            yield current_frame + step * closest_frame
        current_frame_id = closest_frame_id
        while True:
            current_frame_id = (current_frame_id + 1) % len(self.all_frames)
            yield self.all_frames[current_frame_id]


class WalkCommand(Enum):
    IDLE = 0
    FORWARD = 1
    LEFT = 2
    BACK = 3
    RIGHT = 4
    LEFT_TURN = 5
    RIGHT_TURN = 6
    DANCE = 7


class UnifiedFixedWalkController:
    """
    Stores a set of fixed walk cycles, and then returns/interpolates between them based on incoming commands. This
    allows for easy switching between walk cycles, so we can send one command to make the robot move left, and the send
    another to make it transition to moving back, for example.
    """

    DEFAULT_WALK_CYCLE_MAP = {
        WalkCommand.IDLE: "WalkConfigs/simple_walk_idle_config.yaml",
        WalkCommand.FORWARD: "WalkConfigs/simple_walk_forward_config.yaml",
        WalkCommand.LEFT: "WalkConfigs/simple_walk_left_config.yaml",
        WalkCommand.BACK: "WalkConfigs/simple_walk_back_config.yaml",
        WalkCommand.RIGHT: "WalkConfigs/simple_walk_right_config.yaml",
        WalkCommand.LEFT_TURN: "WalkConfigs/simple_walk_left_turn_config.yaml",
        WalkCommand.RIGHT_TURN: "WalkConfigs/simple_walk_right_turn_config.yaml",
        WalkCommand.DANCE: "WalkConfigs/dance_config.yaml",
    }

    def __init__(
        self, speed=1.0, command_to_walk_cycle_config_map=None, initial_position=None
    ):
        if command_to_walk_cycle_config_map is None:
            command_to_walk_cycle_config_map = self.DEFAULT_WALK_CYCLE_MAP
        self.command_to_walk_cycle = {
            command: WalkCycle(walk_cycle_config, speed=speed)
            for command, walk_cycle_config in command_to_walk_cycle_config_map.items()
        }
        self.previous_command = None
        self.current_walk_cycle_generator = None
        if initial_position is None:
            self.current_frame = np.asarray([0.5 for _ in range(MAX_SERVO_COUNT)])
        else:
            self.current_frame = initial_position

    def get_next_step(self, command: WalkCommand):
        """
        Moves along one of the walk commands if the command doesn't change. If it changes, transition over to the new
        walk cycle by interpolating with the closest position.
        :param command: a walk command which is a key in the walk command map
        :return: a frame of servo positions
        """
        if command != self.previous_command:
            self.current_walk_cycle_generator = self.command_to_walk_cycle[
                command
            ].start_from_position(self.current_frame)
            self.previous_command = command
        self.current_frame = next(self.current_walk_cycle_generator)
        return self.current_frame
