import time

import numpy as np
import serial


class ArduinoSerial():
    MAX_PULSE = 3000
    MIN_PULSE = 600
    MAX_SERVO_COUNT = 8
    PULSE_GRANULARITY_16BIT = 2

    def __init__(self, port: int, baudrate: int = 9600):
        self.ser = serial.Serial(port, baudrate=baudrate)
        self.ser.timeout = 0.024
        self.to_send = {}

    @staticmethod
    def centre() -> int:
        return int((ArduinoSerial.MAX_PULSE + ArduinoSerial.MIN_PULSE) / 2)

    def send(self):
        data = [self.get_16bit_servo_command(servo_id, position) for servo_id, position in self.to_send.items()]
        for command in data:
            self.ser.write(command)

    @staticmethod
    def get_16bit_servo_command(servo_id: int, position: int) -> bytes:
        np.clip(position, ArduinoSerial.MIN_PULSE, ArduinoSerial.MAX_PULSE)
        assert servo_id < ArduinoSerial.MAX_SERVO_COUNT, 'Error -- Servo ID exceeds max servo count'
        discretized_position = int(position / ArduinoSerial.PULSE_GRANULARITY_16BIT)
        command = (discretized_position << 3) + servo_id
        assert command >> 14 == 0, 'Error command exceeds 16 bits'
        second_byte = (command & 127) | 128
        first_byte = ((command >> 7) & 127)
        return bytes([first_byte, second_byte])

    def get_data(self) -> dict:
        raw_data = self.ser.read_all()

        print(raw_data)

    def close_ports(self):
        self.ser.close()


if __name__ == '__main__':
    arduino_controller = ArduinoSerial('/dev/ttyUSB0')
    try:
        i = 0
        while True:
            i += 1
            arduino_controller.to_send = {7: 2400}
            arduino_controller.send()
            data = arduino_controller.get_data()
            time.sleep(0.05)

    finally:
        arduino_controller.close_ports()
        print('ports closed')
