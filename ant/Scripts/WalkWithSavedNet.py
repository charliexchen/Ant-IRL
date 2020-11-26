#!/usr/bin/python
import pygame
import haiku as hk
import jax
import time
import numpy as np
import pickle
from ServoController.WalktCycleConfigParser import WalkCycle
from ServoController.SerialServoController import SerialServoController


def net(current_position):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.sigmoid,
        hk.Linear(256), jax.nn.sigmoid,
        hk.Linear(8), jax.nn.sigmoid
    ])
    return (mlp(current_position) * 2) - 1


rng = jax.random.PRNGKey(42)

wc = WalkCycle()
input_dataset = wc.get_training_data()
current_position, label = next(input_dataset)
commands = wc._get_frames()
steps = [next(commands), next(commands), next(commands)]
print(steps)
net_t = hk.transform(net)
net_t.init(rng, current_position)
params = pickle.load(open("NetConfigs/params_interrupted.p", "rb"))

evaluate = jax.jit(net_t.apply)

arduino_controller = SerialServoController('/dev/ttyUSB1')
try:
    while True:
        current_position = evaluate(params, None, np.concatenate(steps))
        print(WalkCycle.frame_to_command(current_position))
        arduino_controller.send(WalkCycle.frame_to_command(current_position))
        steps.append(current_position)
        steps.pop(0)
        print(steps)
        time.sleep(0.05)
finally:
    arduino_controller.send_centre_command()
    arduino_controller.close_ports()
    print('ports closed')
