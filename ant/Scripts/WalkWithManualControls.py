#!/usr/bin/python
import operator

import pygame
from pygame.locals import *

from ServoController.SerialServoController import SerialServoController
from ServoController.WalktCycleConfigParser import (
    UnifiedFixedWalkController,
    WalkCommand,
)

pygame.init()
clock = pygame.time.Clock()

pygame.display.set_mode()
arduino_controller = SerialServoController("/dev/ttyUSB0")
walk_cycle_controller = UnifiedFixedWalkController(0.5)

key_to_command_map = {
    K_w: WalkCommand.FORWARD,
    K_a: WalkCommand.LEFT,
    K_s: WalkCommand.BACK,
    K_d: WalkCommand.RIGHT,
    K_q: WalkCommand.LEFT_TURN,
    K_e: WalkCommand.RIGHT_TURN,
    K_SPACE: WalkCommand.DANCE,
}

frame_counter = 0
default_to_idle_counter = 0
keyboard_commands = {}
try:
    while True:
        frame_counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_command_map:
                    keyboard_commands[key_to_command_map[event.key]] = frame_counter
            if event.type == pygame.KEYUP:
                if event.key in key_to_command_map:
                    keyboard_commands.pop(key_to_command_map[event.key])
        keyboard_command = None
        if len(keyboard_commands) == 0:
            default_to_idle_counter -= 1
            if default_to_idle_counter < 0:
                keyboard_command = WalkCommand.IDLE
        else:
            keyboard_command = max(
                keyboard_commands.items(), key=operator.itemgetter(1)
            )[0]
            default_to_idle_counter = 10
        print(keyboard_command)
        if keyboard_command is not None:
            serial_command = walk_cycle_controller.get_next_step(keyboard_command)
            arduino_controller.send({id: com for id, com in enumerate(serial_command)})
        clock.tick(20)
finally:
    arduino_controller.send_idle_command()
    arduino_controller.close_ports()
