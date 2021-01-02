import yaml
from AntController.AntEnvironment import AntIRLEnvironment
from AntController.HaikuPredictor import HaikuPredictor
from AntController.HaikuActorCritic import HaikuContinuousActorCritic

if __name__ == "__main__":
    path = "AntController/configs/ant_aac_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    aac = HaikuContinuousActorCritic(config)
    env = AntIRLEnvironment()
    for _ in range(100):
        aac.run_episode_and_train(env, True, True)
    env.reset()