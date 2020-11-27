#!/usr/bin/python
import pygame
from pygame.locals import *
from ServoController.SerialServoController import SerialServoController
from ServoController.WalktCycleConfigParser import WalkCycle

pygame.init()
clock = pygame.time.Clock()

pygame.display.set_mode()
arduino_controller = SerialServoController('/dev/ttyUSB0')
walk_cycle_forward = WalkCycle("WalkConfigs/simple_walk_config.yaml").get_commands()
walk_cycle_back = WalkCycle("WalkConfigs/simple_walk_back_config.yaml").get_commands()
walk_cycle_left = WalkCycle("WalkConfigs/simple_walk_left_turn_config.yaml").get_commands()
walk_cycle_right = WalkCycle("WalkConfigs/simple_walk_right_turn_config.yaml").get_commands()
command = {}
current_command = None
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == K_UP:
                    current_command = walk_cycle_forward
                elif event.key == K_LEFT:
                    current_command = walk_cycle_left
                elif event.key == K_DOWN:
                    current_command = walk_cycle_back
                elif event.key == K_RIGHT:
                    current_command = walk_cycle_right
            if event.type == pygame.KEYUP:
                current_command = None
        if current_command:
            command = next(current_command)
            arduino_controller.send(command)
        clock.tick(20)
finally:
    arduino_controller.send_centre_command()
    arduino_controller.close_ports()
