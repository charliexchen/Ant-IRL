# Ant-IRL

Ant-v2 (A.K.A Antony) is now a fairly standard RL task from the Open AI gym library. This project aims to achieve three goals:
* Build a physical robot in a similar configuration to Ant-v2
* Train Actor Critic on a physical environment with no simulation
* Use Haiku with JAX (https://github.com/deepmind/dm-haiku) on an end to end project

...in addition to staying sane over the third lock down in the UK.
 <p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/walk_comparison.gif" align="centre" width="1000" >
 </p>
<p align="center"><i> <sub>Improvements in the robot's speed after training with AAC with experience replay in the simplified environment. Left: pretrained agent with some random noise in actions. Middle: agent after optimising for 33 episodes. Right: agent after optimising for 91 episodes</sub></i> </p>


## Building the Robot

<p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/assem.png" width="1000">
</p>

First, the robot was designed using Siemens Solid Edge CE (a powerful but free CAD software) and then 3D printed using my Creality Ender 3. The design consists of 8 servos in a similar configuration to Ant-v2 (albeit the forelegs are shorter to reduce the load on the tiny 9g servos). For control, I used an Arduino Nano which communicates with my desktop via USB serial and the servo controller via i2c. The position of each servo can be manipulated by sending two bytes of data.


For sensing, the Arduino is also connected to a gyro/accelerometer via i2c, which gives us acceleration, the gravity vector and Euler angles. Using the MPU-6050 sensor board's onboard DMP feature, it is not necessary to implement further noise reduction (such as Kalman/complement filters) for the sensors. As a future upgrade, I have designed micro switch holders for the forelegs which will allow the robot to know if the legs have contacted the ground.

The robot runs on a 5v 2A DC power supply. Power and USB connection is maintained using 6 thin enamelled copper wires (usually used for electromagnets). This minimises any forces a stiff USB/power cable might impart onto the robot.

## Location Detection

The Robot's location and orientation relative to the environment is detected via the large aruco marker on top of the robot and markers on the corners of the environment. This was achieved with the aruco module in OpenCV. Under the correct lighting conditions, we can have the location of the robot over 99% of the time, and we can resort to last frame position or interpolating for any missing frames.

<p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/camera.png" width="1000">
</p>
<p align="center"><i> <sub>Using OpenCV to correct the raw webcam input. Note the small amount of parallax if the camera is not directly overhead. This is usually small enough to not affect the results.</sub></i></p>

The capture setup consists of a cheap webcam on an angled tripod, pointing downwards. With the locations of the corners of the environment known, perspective and fisheye distortion can be corrected with standard OpenCV operations.

<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/walk.gif" align="centre" width="400" ></a>
</p>
<p align="center"><i> <sub>Camera setup and simple walk. Note the trailing wire connected to power and USB.</sub></i></p>

## Building an Environment

With the above setup, we can implement a state-action-reward loop, which forms the basis of the MDP. The robot would walk forth, collect data, train on that data using one's RL algorithm of choice, before resetting. 

* State consists of sensor inputs, absolute position, orientation, servo positions, creating a 22-dimensional vector (The simplified version of the environment removes sensor inputs, reducing the dimensionality to 12)
* Actions are 8 dimensional vectors representing either new absolute servo positions or deltas between the current position to the next.
* Reward is based on position, with +2 if it reaches the end of the field, -1 if it goes too far off the side and up to 1 for moving from left to right. The simplfied version of the envionment has the velocity dotted with the normalised orientation vector as the reward (essentially the forward speed).

A hand engineered walk cycle is also implemented. This was done for three reasons:
* Given a target and the robot's position, you can let the robot move up/down/left/right until it reached the target. This gives us a reliable method of resetting the robot's position after running an episode.
* We can collect data on this fixed walk cycle (in particular accelerometer/gyro data) in order to pretrain policiy/value functions.
* This makes room for lots of fun, future, non deep-learning projects. I can simply map the commands to keyboard/some other controller, and we basically have a walker which can reliably move in any direction.

<p align="center">
  <a href="url"><img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/manual_walk.gif" align="centre" width="400" ></a>
</p>
<p align="center"><i> <sub>Walking to specified locations by combining the camera and the hand engineered gait. This is used to reset the environment.</sub></i></p>

## Implementing and Running AAC

I implemented Advantage Actor-Critic using JAX and Haiku. This was a fun framework to use since it is a lot more flexible compared to the others I've tried in the past, such as Keras. Almost any numpy operation can be accelerated, vectorised or differentiated, so I can see this handling more unconventional architectures much more gracefully, even though the overall feature gap isn't huge compared to other frameworks. For example, I especially liked how for autograd, we use grad which is treated like a function operator much like how gradients are represented mathematically.

I made the predictor and AAC class as generic as possible for future projects, and tested that it works on CartPole.

Since this is a physical environment with one agent, we are very data constrained. As such, I first ran a number of episodes using the fixed walk cycle. We collect this data and then use it to pretrain the value function and an actor based on a fixed gait. We then use the above AAC implementation along with experience replay to improve the agent. The initial data also allows for some hyperparameter selection, which was done with a simple random search. 

 <p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/env_walk.gif" align="centre" width="400" >  
 </p>
<p align="center"><i> <sub>Running multiple episodes with fixed agent in the environment to collect training data. Red rectangles terminate with negative reward, green with positive reward. After an episode, the agent returns to the red circle to reset the environment. Note the aruco markers used to ensure consistent perspective in the corners of the environment. </sub></i> </p>

Other things to note about the AAC implementation:
* The environment runs at 30fps (bounded by the frame rate of the camera). This can be downsampled by simply issuing a new action only every x frames. This can give the NNs a longer effective time horizon, in addition to making the setup less sensitive to latency in the camera, sensors and servo controller.
* The problem can be further simplified by imposing symmetries into the robot's actor, though this does remove the robot's ability to turn.
* In order to learn a continuous action space, the actor simply returns the mean values of a normal distribution. The variance is fixed for exploration, but it can also be the output of the actor.

<p float="left" align="center">
  <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/crit_loss.png" width="250" />
 <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/act_loss.png" width="250" />
 <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/episodic_reward.png" width="250" />
</p>
<p align="center"><i> <sub>Fig 1: loss function at each episode of the value critic. Given AAC is on-policy, we can expect this value to not go down as long as the policy has not converged. Fig 2: objective of the actor policy. This is the log-likelihood scaled advantage, and so we expect it to go up as the agent improves. Fig 3: Episodic cumulative reward -- the agent is moving faster as the policy improves.</sub></i> </p>

Tuning RL algorithms is hard. To make my life easier, I followed the best some best practices:
* Normalise the input and ouput of the NNs so the values across the layers are of similar magnitude
* Offline training/hyperparameter tuning uses train/dev/test sets to prevent overfitting
* Experiment configurations are separated into config files
* End to end tests for all major components in order to ensure correctness

## References
Main papers I consulted:
* [Challenges of Real-World Reinforcement Learning](https://arxiv.org/abs/1904.12901) -- This is a deepmind paper which gave a high level overview of the kind of problems I'm going to encounter, and served as a good jumping off point.
* [Automated DeepReinforcement Learning Environment for Hardware of a Modular Legged Robot](https://ieeexplore.ieee.org/document/8442201) -- A Disney research paper on getting various configurations to walk using DDPG and TRPO on a modular walking robot. Pretty cool, and gave me an idea about possible RL algorithm candidates. This paper actually trained the agent starting with one leg and increasing the number as legs as you go.
* [Optimizing walking of a humanoid robot using reinforcement learning](https://www.researchgate.net/publication/267024754_Reinforcement_Learning_with_Experience_Replay_for_Model-Free_Humanoid_Walking_Optimization) -- A masters thesis from university of warsaw using AAC. My architecture ended up being pretty similar to this one.
* [RealAnt: An Open-Source Low-Cost Quadruped for Research in Real-World Reinforcement Learning](https://arxiv.org/abs/2011.03085) -- A paper which also builds a open source robot, and trains some gaits on it. This was published days after I had done most of CAD work, so that was a little annoying. However, this design used much more expensive servos (each servo cost more than my robot put together) which has inbuilt sensors, superior control and full 360 rotation, and I think that might be worth investing in...
## Parts List
* [Camera](https://www.amazon.co.uk/gp/product/B08CXV11MX/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1) -- Basically any webcam will do, though I ended up using this one side it comes with a standard standard 1/4-20 threaded mount for use on a tripos, in addition to a hinge and ball mount. These features are fairly common in other webcams though, so don't worry about getting the exact one.
* [Arduino Nano](https://www.amazon.co.uk/ELEGOO-Arduino-board-ATmega328P-compatible/dp/B072BMYZ18/ref=sr_1_1_sspa?dchild=1&keywords=arduino+nano&qid=1619280156&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzNUw0RjJHVFlOWDBCJmVuY3J5cHRlZElkPUEwMDA0OTIxMjA2VTY1V0NUQzlLOCZlbmNyeXB0ZWRBZElkPUEwODQ2MTQwM1RJMzE1WFlLODEzMyZ3aWRnZXROYW1lPXNwX2F0ZiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU=) -- Standard arduino. [There are lots of tutorials online for using these](https://store.arduino.cc/).
* [PCA9685 Servo Controller](https://www.amazon.co.uk/TeOhk-Channel-Interface-Controller-Raspberry/dp/B07RVJRQX1/ref=sr_1_1_sspa?dchild=1&keywords=PCA9685&qid=1619278824&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzSUNGMUZXTU8wTTRZJmVuY3J5cHRlZElkPUEwNzU3MjAzTkc3QVI5VFhaRkFPJmVuY3J5cHRlZEFkSWQ9QTAxMDkyMDIzUFZPVlI3WDZBM0pNJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==) -- Interfaces with arduino with i2c and connects to external power.
* [MPU-6050 Sensor Board](https://www.amazon.co.uk/Neuftech%C2%AE-MPU-6050-3-gyroscope-accelerometer-crafting/dp/B00PIMRJX6/ref=sr_1_1_sspa?dchild=1&keywords=Accelerometers&qid=1619280233&s=electronics&sr=1-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUExN1RTSDhNV0ZaSUgmZW5jcnlwdGVkSWQ9QTA1NTAyNTQzTkFYOUlYNFpLSVhNJmVuY3J5cHRlZEFkSWQ9QTA1ODE4NTkzREdRTkFLU1U0MDlNJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==) -- Accelerometer, gyro and temperature sensor on a single board which can then communicate with i2c. Gyros are very accurate, but its values need to be integrated in order to get angular position, resulting in drift. Accelerometers don't have problems in drift, but are often really noisy. It is possible to combine data from both sensors in order to get the best of both worlds, and the DMP feature on these boards mean you don't have to write the complement filter yourself. Tutorial on how to get the DMP working can be found [here](https://maker.pro/arduino/tutorial/how-to-interface-arduino-and-the-mpu-6050-sensor).
* [Tower Pro 9 Servos](https://www.amazon.co.uk/gp/product/B07LBS7WYZ/ref=ppx_yo_dt_b_asin_title_o03_s00?ie=UTF8&psc=1) -- Super cheap 9g servos used in model aircraft. I recommend looking at some more expensive ones with the same dimensions however, since these were pretty imprecise.
* [SUNLU PLA filament](https://www.amazon.co.uk/gp/product/B07TMKGGS3/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1) -- I used this black filament, but PLA is so forgiving a plastic to print it really doesn't matter which brand you use.
* [0.2mm Enamelled copper wire](https://www.ebay.co.uk/itm/143728062658?ssPageName=STRK%3AMEBIDX%3AIT&_trksid=p2060353.m2749.l2649) -- Usually used for making electromagnets. I used this to make a long flexible cable with 6 wires, 4 for USB, and 2 for powering the servos. Some light soldering is needed here.
* [5v 2a power supply](https://www.amazon.co.uk/gp/product/B08DXX6LGG/ref=ppx_yo_dt_b_asin_title_o06_s00?ie=UTF8&psc=1) -- Used to power the servos. You can power one or two servos directly with the arduino, but you'll need an external power supply if you want to power more -- drawing too much current will damage the arduino. Be careful with these -- whilst they're not going to burn your house down, they're surprisingly expensive compared to the rest of the project and will fry themselves if shorted. If you forsee that you're going to be doing a lot of electrionic projects, I recommend getting one of these [variable DC power supplies](https://www.amazon.co.uk/gp/product/B08HQG9D56/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1), since they're much more flexible and is current limited, protecting it against shorts.
<p align="center">
   <img src="https://github.com/charliexchen/Ant-IRL/blob/main/Assets/readme_assets/ant.png" align="centre" width="400" >  
 </p>
 <p align="center"><i> <sub>You've made it to the end! Thank you for reading! </sub></i> </p>
