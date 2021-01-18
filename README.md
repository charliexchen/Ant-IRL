# Ant-IRL

Ant-v2 (A.K.A Antony) is now a fairly standard RL task from the Open AI gym library. Since I'm stuck in lockdown, here is a project trying to bring him/her to real life. If all goes well, we can then try to train Actor-Critic on this environment.

 <p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/env_walk.gif" align="centre" width="400" >  
 </p>
<p align="center"><i> <sub>Running multiple episodes with fixed agent in the environment to collect training data. Red rectangles terminate with negative reward, green with positive reward. After an episode, the agent returns to the red circle to reset the environment. Note the aruco markers used to ensure consistent perspective in the corners of the environment. </sub></i> </p>

This also allows me to test out Haiku with JAX (https://github.com/deepmind/dm-haiku), which is a relatively new ML framework used in Google Deepmind which promises more flexibility.

## Building the Robot

First, the robot was designed using CAD (Siemens Solid Edge) and then 3D printed using an Ender 3. The design consists of 8 servos configured similarly as Ant-v2 (albeit the forelegs are shorter to reduce the load on the tiny 9g servos). For control, I used an Arduino Nano which communicates with the PC via USB serial and the servo controller via i2c, and the position of each servo can be manipulated by sending two bytes of data.

<p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/ant.png" width="400">
</p>
<p align="center"><i> <sub>CAD model of the robot in Solid Edge. Parts are then sliced and 3D printed.</sub></i> </p>

For sensing, the Arduino is also connected to a gyro/accelerometer using i2c, which gives us acceleration, the gravity vector and Euler angles. Using the MPU-6050's onboard DMP feature, it is not necessary to implement further noise reduction (such as Kalman filters) for the sensors. As a future upgrade, I have designed micro switch holders for the forelegs which will allow the robot to know if the legs have contacted the ground.

<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/ant_irl.png" align="centre" width="300" ></a>
</p>
<p align="center"><i> <sub>"Don't talk to me or my son ever again."</sub></i> </p>

The robot runs on a 5v 2A DC power supply. Power and USB connection is maintained via 6 thin enamelled copper wires to reduce cable imparting forces on the environment.

## Location Detection

The Robot's location and orientation relative to the environment is detected via the large aruco marker on top of the robot and markers on the corners of the environment. This was achieved with the aruco module in OpenCV. Under the correct lighting conditions, we can have the location of the robot over 99% of the time, and we can resort to last frame position or interpolating for any missing frames steps.

The capture setup is simply a cheap webcam on an angled tripod, pointing downwards. With the locations of the corners of the environment, perspective and fisheye distortion can be corrected with standard OpenCV operations.

<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Parts/walk.gif" align="centre" width="400" ></a>
</p>
<p align="center"><i> <sub>Camera setup and simple walk. Note the trailing wire connected to power and USB.</sub></i></p>

## Building an Environment

With the above setup, we can implement a state-action-reward loop, which forms the basis of an RL environment. The robot would walk forth, collect data, train on that data and repeat the process using the RL algorithm of choice. 

* State consists of sensor inputs, absolute position, orientation, servo positions, creating a 22-dimensional vector
* Actions are 8 dimensions, consisting of either new servo positions or deltas between the current position to the next
* Reward is based on position, with +2 if it reaches the end of the field, -1 if it goes too far off the side and up to 1 for moving from left to right.

In order to reset the environment, a simple fixed walk cycle loop is implemented. This allows the robot to walk itself back to the starting position. We can also collect data on this fixed walk cycle in order to pretrain policies/value functions.

(With the fix walk cycle, the robot can walk in 4 cardinal directions and turn with keyboard commands -- This makes room for lots of fun future projects :D)

## Implementing and Running AAC

I implemented Advantage Actor-Critic using JAX and Haiku. This was a fun framework to use since it is a lot more flexible compared to the others I've tried in the past, such as Keras. Almost any numpy operation can be accelerated, vectorised or differentiated, so I can see this handling more unconventional architectures much more gracefully, in addition to any other high-speed operations.

I tried making the predictor and AAC class as generic as possible for future projects and tested that it works on CartPole.

Since this is a physical environment with one agent, we are very data constrained. As such, I first ran a number of episodes using the fixed walk cycle. We collect this data and then use it to pretrain the value function and an actor based on a fixed gait. We then use the above AAC implementation along with experience replay (TBD) to improve the agent. The initial data also allows for some hyperparameter selection, which was done with a simple random search. 

