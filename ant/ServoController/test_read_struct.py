import time
from struct import *

import serial

ser = serial.Serial(
    port='/dev/ttyUSB1',
    baudrate=9600
)

size = calcsize('fffffffffff')

buffer = []


def get_latest_from_buffer(buffer, latest=[], separator=[0, 252, 127, 63], format='=fffffffffff'):
    size = calcsize(format)
    new_buffer = []
    i = 0
    if len(buffer) < size:
        return buffer, latest
    while i < len(buffer):
        matched = True
        for j, sub_ele in enumerate(separator):
            if i + j >= len(buffer) or sub_ele != buffer[i + j]:
                matched = False
            break
        if matched:
            if len(buffer) - i >= size:
                latest = buffer[i:i + size]
                i += size
            else:
                new_buffer = buffer[i:]
                break
        else:
            i += 1
    return new_buffer, latest


buff = []
latest = []
while True:
    buff.extend(ser.read_all())
    buff, latest = get_latest_from_buffer(buff, latest)
    if len(latest) >= 44:
        print(unpack("fffffffffff", bytes(latest))[5])
    time.sleep(0.5)
