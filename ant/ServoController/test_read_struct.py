import serial, time
from  copy import copy
from struct import *


ser = serial.Serial(
  port='/dev/ttyUSB0',
    baudrate=9600
)

size = calcsize('fffffffffff')

buffer = []
def split_substring():
    pass
while True:
    print(list(ser.read_all()))
    #print(list(data_), len(data_))
    #while ser.in_waiting > size:
    #    data_ = ser.read(size)
    #ser.flushInput()
   # if data_:
   #     print(list(data_[:4]))


    #if data_:
    #    print(unpack('ffffffff', data_)[4:])
    time.sleep(0.03)