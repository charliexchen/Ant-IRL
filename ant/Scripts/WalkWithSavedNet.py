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
params = pickle.load(open("configs/Ant/params_89162_gen.p", "rb"))

evaluate = jax.jit(net_t.apply)

arduino_controller = SerialServoController('/dev/ttyUSB0')
try:
    while True:
        current_position = evaluate(params, None, current_position)
        current_position = np.random.normal(current_position, 0.5)
        arduino_controller.send(WalkCycle.frame_to_command(current_position))
        time.sleep(0.05)
finally:
    arduino_controller.send_idle_command()
    arduino_controller.close_ports()
    print('ports closed')
