#!/usr/bin/python
import time
import numpy as np
import serial
from ServoController.ServoIDConfig import MAX_PULSE, MIN_PULSE, MAX_SERVO_COUNT, is_inverted
from ServoController.WalktCycleConfigParser import WalkCycle


class SerialServoController:
    PULSE_GRANULARITY_16BIT_8SERVOS = 1

    def __init__(self, port: int, baudrate: int = 9600):
        self.ser = serial.Serial(port, baudrate=baudrate)
        self.ser.timeout = 0.024
        self.current_command = {servo_id: self.centre() for servo_id in range(MAX_SERVO_COUNT)}

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

    def get_data(self) -> dict:
        raw_data = self.ser.read_all()
        return raw_data

    def close_ports(self):
        self.ser.close()


if __name__ == '__main__':
    arduino_controller = SerialServoController('/dev/ttyUSB1')
    walk_cycle = WalkCycle("WalkConfigs/simple_walk_left_turn_config.yaml").get_commands()
    try:
        i = 0
        while True:
            i += 1
            arduino_controller.send(next(walk_cycle))
            data = arduino_controller.get_data()
            time.sleep(0.05)
    finally:
        arduino_controller.send_centre_command()
        arduino_controller.close_ports()
        print('ports closed')
