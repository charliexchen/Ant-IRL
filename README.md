# Ant-IRL

Ant(ony)-v2 is now a fairly standard RL task from the Open AI gym library. This is a project to try to bring him to real life whilst I'm stuck at home due to lock down, and then try to train Actor Critic on th environment.

<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/env_walk.gif" align="centre" width="400" ></a>
  </p>

This also gives me the opportunity to test out Haiku (https://github.com/deepmind/dm-haiku), which is a a relatively new ML library used in Google Deepmind which promises more flexibility compared to many other methods.

## Building the Robot
  <p align="center">
    <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/ant.png" width="400">
  </p>

First, the robot was designed using CAD (Siemens Solid Edge) and then 3D printed. It consists of 8 servos configured in a similar manner as Ant-v2. For control, I used an arduino nano which communicates with the PC via USB serial and the servo controller via i2c, and the position of each servo can be manipulated by sending two bytes of data.

For sensing, the arudino is also connected to a gyro/accelerometer using i2c, which gives us acceleration, the gravity vector and euler angles. Using the MPU-6050's onboard DMP feature, it is not necessary to implement further noise reduction (such as kalman filters) for the sensors. As a future upgrade, I have designed micro switch holders for the forelegs which will allow the robot to know if the legs have contacted the ground.
<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/ant_irl.png" align="centre" width="400" ></a>
  </p>

## Location Detection

The Robot's location and orientation relative the environment is detected via the aruco markers on the robot and and the corners of the environment. Most of this was achieved with standard libraries within OpenCV. Under the correct lighting conditions we can have the location of the robot over 99% of the time, and we can resort to last frame position or interpolating for the missing frames steps.
<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/walk.gif" align="centre" width="400" ></a>
  </p>

## Building an Environment

With the above setup, we can implement an state-action-reward loop, which forms the basis of an RL environment. The robot would walk forth, collect data, train on that data dn repeat the proccess using the RL algorithm of choice. 

Reward is based on position, with +2 if it reaches the end of the field, -1 if it goes too far off the side and up to 1 for moving from left to right.

In order to reset the environment, a simple fixed walk cycle loop is implemented. This allows the robot to walk itself back to the starting position (and also for manual control in future projects). We can also collect data on this fixed walk cycle in order to pretrain policies/value functions, which would allieviate the data-constraints.

## Training on the Environment

I implemented Actor Critic using JAX and Haiku. 

A simple random search was implemented

