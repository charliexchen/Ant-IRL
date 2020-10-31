# Ant-IRL

Ant(ony) is now a fairly standard RL task from the Open AI gym library. For fun, let's try to bring him to real life.

Here is the Plan:
1) train actor-critic on Open AI gym in order to get the code working.
2) 3d print and build Ant(ony).
3) build classical control/fixed loop control system using raspberry pr zero/arduino nano.
4) build a test environment.
5) collect data from the test environment, with the classical control setup, collecting sensor data, and initialise the models using this data.
6) Start collecting data from that which is generated by the model, and optimise the ant's walk using a walk cycle.
