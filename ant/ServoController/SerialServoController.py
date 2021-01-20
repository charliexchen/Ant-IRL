#!/usr/bin/python
import numpy as np
import serial
import struct
import time

from ServoController.ServoIDConfig import (
    MAX_PULSE,
    MIN_PULSE,
    MAX_SERVO_COUNT,
    is_inverted,
)
from ServoController.WalktCycleConfigParser import WalkCycle


class SerialServoController:
    """
    Class which sends and receives commands to/from the arduino via pySerial. The Arduino parses the commands and moves
    the robot. We can parse the serial from the robot, and this way we can access the accelerometer data.
    """

    PULSE_GRANULARITY_16BIT_8SERVOS = 1
    DEFAULT_SEPARATOR = [0, 252, 127, 63]  # 100000000.0 float value
    CALIBRATION = [0, 0, 0, 0, -50, -100, -100, 0]

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        input_data_format: str = "fffffffffff",
        reset_threshold=50,
    ):
        print("Connecting to Antony via serial...")
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate=baudrate)
        self.ser.timeout = 0.024
        self.current_command = {
            servo_id: self.centre() for servo_id in range(MAX_SERVO_COUNT)
        }
        self.struct_format = input_data_format
        self.struct_size = struct.calcsize(input_data_format)
        self.buffer = []
        self.latest = []
        self.reset_threshold = reset_threshold
        self.reset_counter = 0
        self.ready = False

        print("Waiting for Antony to respond...")
        self.wait_for_connection()

    @staticmethod
    def _get_16bit_servo_command(servo_id: int, position: int) -> bytes:
        """
        Converts a servo command pair into two bytes, which can then be sent to the robot
        :param servo_id
        :param position: servo command based on PWM
        :return: two bytes which the arduino can parse and act on.
        """
        calibrated_pos = position + SerialServoController.CALIBRATION[servo_id]
        calibrated_pos = np.clip(calibrated_pos, MIN_PULSE, MAX_PULSE)
        assert servo_id < MAX_SERVO_COUNT, "Error -- Servo ID exceeds max servo count"
        discretized_position = int(
            (calibrated_pos - MIN_PULSE)
            / SerialServoController.PULSE_GRANULARITY_16BIT_8SERVOS
        )
        command = (discretized_position << 3) + servo_id
        assert command >> 14 == 0, "Error command exceeds 16 bits"
        second_byte = (command & 127) | 128
        first_byte = (command >> 7) & 127
        return bytes([first_byte, second_byte])

    def _get_command_from_normalised_positions(self, servo_id, normalised_position):
        """
        Converts float from +/-1 to servo command. +1 means legs move forward, and feet moves down. This makes it easier
        for the ML models later on
        :param servo_id
        :param normalised_position: position of servo from +/-1, raw ML model output
        :return: servo command based on PWM
        """
        if is_inverted(servo_id):
            normalised_position *= -1
        return self.centre() + int(normalised_position * ((MAX_PULSE - MIN_PULSE) / 2))

    @staticmethod
    def centre() -> int:
        return int((MAX_PULSE + MIN_PULSE) / 2)

    def send(self, commands):
        """
        Sends the command to the robot.
        :param commands: dict of servo to position
        """
        if not self.ready:
            return
        for servo_id in commands:
            self.current_command[
                servo_id
            ] = self._get_command_from_normalised_positions(
                servo_id, commands[servo_id]
            )
        commands = [
            self._get_16bit_servo_command(servo_id, position)
            for servo_id, position in self.current_command.items()
        ]
        for command in commands:
            self.ser.write(command)

    def send_centre_command(self):
        self.send({servo_id: 0 for servo_id in range(MAX_SERVO_COUNT)})

    def send_idle_command(self):
        """
        Sends a single command which puts the robot to the idle position.
        """
        self.send({0: 0, 1: 0, 2: 0, 3: 0, 4: 0.9, 5: 0.9, 6: 0.9, 7: 0.9})

    def send_pack_up_command(self):
        """
        Sends a single command which makes the robot curl up into a ball so I can pack it snugly in a box.
        """
        self.send(
            {0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: -1.0, 7: -1.0}
        )

    def get_data_if_ready(self):
        """
        Takes the raw data from the arduino, and parses it so we can access the sensor information. The loaded data is
        stored in self.latest, so is no data is received we just send the same as the last frame.
        :return: parsed struct based on input_data_format, None if the robot has frozen or the system is not yet ready.
        """
        serial_data = self.ser.read_all()
        if serial_data == b"Ant Ready!\r\n":
            print("Ready!")
            self.ready = True
        if not self.ready:
            return
        if self._detect_frozen_state(serial_data):
            return
        self.buffer.extend(serial_data)
        self._update_serial_data()
        if len(self.latest) == self.struct_size:
            return struct.unpack(self.struct_format, bytes(self.latest))

    def _detect_frozen_state(self, serial_data):
        """
        Checks if the accelerometer is returning nothing for too long, and resets the connection if it happens.
        :return: Boolean if frozen state was detected
        """
        if len(serial_data) == 0:
            self.reset_counter += 1
            if self.reset_counter > self.reset_threshold:
                print("Robot is frozen -- resetting the connection...")
                self.reset_connection()
                return True
        else:
            self.reset_counter = 0
        return False

    def _update_serial_data(self):
        """
        Ugly piece of code which splits the data in the buffer based on the separator bytes and saves it into
        self.latest. Since the arduino is sending raw bytes, we can't have a dedicated separator byte, so we have a
        sequence to reduce the prob of a collision.
        :return: void
        """
        if not self.ready:
            return
        i = 0
        if len(self.buffer) < self.struct_size:
            return
        while i < len(self.buffer):
            matched = True
            for j, sub_ele in enumerate(self.DEFAULT_SEPARATOR):
                if i + j >= len(self.buffer) or sub_ele != self.buffer[i + j]:
                    matched = False
                break
            if matched:
                if len(self.buffer) - i >= self.struct_size:
                    self.latest = self.buffer[i : i + self.struct_size]
                    i += self.struct_size
                else:
                    self.buffer = self.buffer[i:]
                    break
            else:
                i += 1

    def close_ports(self):
        self.ser.close()

    def reset_connection(self):
        """
        The DMP class on the arduino is not very stable, and occasionally the robot freezes. This closes and resets the
        connection, which would cause the robot to reset and allow the environment to continue running. This can also
        if the cable gets jostle a little too much.

        We can avoid this by flashing a version of the arduino sketch which doesn't make use of the accelerometer, but
        obviously we lose access to sensor data if that happens.
        """
        self.ser.close()
        self.ser = serial.Serial(self.port, baudrate=self.baudrate)
        self.ser.timeout = 0.024
        self.ready = False
        self.buffer = []
        self.latest = []
        self.reset_counter = 0
        # self.wait_for_connection()

    def wait_for_connection(self):
        while not self.ready or self.latest == []:
            self.get_data_if_ready()
            time.sleep(0.05)


if __name__ == "__main__":
    """Quick end to end test of the class. Creates a connection, and let the robot walk using a fixed walk cycle and 
    print sensor data. When we close the programme with keyboard interrupt, we close the connection. """
    arduino_controller = SerialServoController("/dev/ttyUSB0")
    walk_cycle = WalkCycle(
        "walk_cycle_configs/simple_walk_left_turn_config.yaml"
    ).get_commands()
    try:
        while True:
            arduino_controller.send_idle_command()
            data = arduino_controller.get_data_if_ready()
            print(data)
            time.sleep(0.05)
    finally:
        arduino_controller.send_idle_command()
        arduino_controller.close_ports()
        print("Serial Connection Closed.")
