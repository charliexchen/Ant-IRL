#!/usr/bin/python
import struct
import time

import numpy as np
import serial

from ServoController.ServoIDConfig import MAX_PULSE, MIN_PULSE, MAX_SERVO_COUNT, is_inverted
from ServoController.WalktCycleConfigParser import WalkCycle


class SerialServoController:
    PULSE_GRANULARITY_16BIT_8SERVOS = 1

    def __init__(self, port: str, baudrate: int = 9600, input_data_format='fffffffffff'):
        self.ser = serial.Serial(port, baudrate=baudrate)
        self.ser.timeout = 0.024
        self.current_command = {servo_id: self.centre() for servo_id in range(MAX_SERVO_COUNT)}
        self.struct_format = input_data_format
        self.struct_size = struct.calcsize(input_data_format)
        self.buffer = []
        self.latest = []

    @staticmethod
    def _get_16bit_servo_command(servo_id: int, position: int) -> bytes:
        position = np.clip(position, MIN_PULSE, MAX_PULSE)
        assert servo_id < MAX_SERVO_COUNT, 'Error -- Servo ID exceeds max servo count'
        discretized_position = int((position - MIN_PULSE) / SerialServoController.PULSE_GRANULARITY_16BIT_8SERVOS)
        command = (discretized_position << 3) + servo_id
        assert command >> 14 == 0, 'Error command exceeds 16 bits'
        second_byte = (command & 127) | 128
        first_byte = ((command >> 7) & 127)
        return bytes([first_byte, second_byte])

    def _get_command_from_normalised_positions(self, servo_id, normalised_position):
        """Converts float from +/-1 to servo command. +1 means legs move forward, and feet moves down"""
        if is_inverted(servo_id):
            normalised_position *= -1
        return self.centre() + int(
            normalised_position * ((MAX_PULSE - MIN_PULSE) / 2))

    @staticmethod
    def centre() -> int:
        return int((MAX_PULSE + MIN_PULSE) / 2)

    def send(self, commands={}):
        for servo_id in commands:
            self.current_command[servo_id] = self._get_command_from_normalised_positions(servo_id, commands[servo_id])
        commands = [self._get_16bit_servo_command(servo_id, position) for servo_id, position in
                    self.current_command.items()]
        for command in commands:
            self.ser.write(command)

    def send_centre_command(self):
        self.send({servo_id: 0 for servo_id in range(MAX_SERVO_COUNT)})

    def send_idle_command(self):
        self.send({0: 0, 1: 0, 2: 0, 3: 0, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5})

    def send_pack_up_command(self):
        self.send({0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: -1.0, 7: -1.0})

    def get_data(self):
        self.buffer.extend(self.ser.read_all())
        self._update_serial_data()
        if len(self.latest) == self.struct_size:
            return struct.unpack(self.struct_format, bytes(self.latest))

    def _update_serial_data(self, separator=[0, 252, 127, 63]):
        i = 0
        if len(self.buffer) < self.struct_size:
            return
        while i < len(self.buffer):
            matched = True
            for j, sub_ele in enumerate(separator):
                if i + j >= len(self.buffer) or sub_ele != self.buffer[i + j]:
                    matched = False
                break
            if matched:
                if len(self.buffer) - i >= self.struct_size:
                    self.latest = self.buffer[i:i + self.struct_size]
                    i += self.struct_size
                else:
                    self.buffer = self.buffer[i:]
                    break
            else:
                i += 1

    def close_ports(self):
        self.ser.close()


if __name__ == '__main__':
    arduino_controller = SerialServoController('/dev/ttyUSB1')
    walk_cycle = WalkCycle("WalkConfigs/simple_walk_left_turn_config.yaml").get_commands()
    try:
        while True:
            arduino_controller.send(next(walk_cycle))
            data = arduino_controller.get_data()
            print(data)
            time.sleep(0.05)
    finally:
        arduino_controller.send_idle_command()
        arduino_controller.close_ports()
        print('ports closed')
