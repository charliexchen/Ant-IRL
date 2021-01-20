import numpy as np
import os
import pickle
from random import randint, choices
from tqdm import trange

from AntController.AntEnvironment import EpisodeData
from HaikuHelpers.HaikuPredictor import HaikuPredictor


def _randint_from_pair(range_pair):
    return randint(range_pair[0], range_pair[1])


def generate_random_mlp_config_from_random(config):
    mlp_config = {"scale": config["scale"], "shift": config["shift"]}
    activation = choices(config["permitted_activations"])[0]
    hidden_layer_size_exp_range = config["hidden_layer_size_exp_range"]
    hidden_layer_count_range = _randint_from_pair(config["hidden_layer_count_range"])
    layers = []
    for _ in range(hidden_layer_count_range):
        hidden_layer_size = 2 ** _randint_from_pair(hidden_layer_size_exp_range)
        layers.append({"type": "linear", "size": hidden_layer_size})
        layers.append({"type": activation})
    layers.append({"type": "linear", "size": config["output_size"]})
    if "output_function" in config:
        layers.append({"type": "output_function"})
    mlp_config["layers"] = layers
    return mlp_config


def generate_predictor_from_random_config(config):
    controller_config = {}
    learning_rate = 10 ** _randint_from_pair(config["learning_rate_exp_range"])
    constant_keys = ["loss", "decay", "input_shape", "name", "rng_key", "scale_config"]
    for key in constant_keys:
        controller_config[key] = config[key]
    controller_config["mlp"] = generate_random_mlp_config_from_random(
        config["mlp_search_config"]
    )
    controller_config["learning_rate"] = learning_rate

    return (
        HaikuPredictor.generate_controller_from_config(controller_config),
        controller_config,
    )


def create_predictor_from_hyper_param_search(config, data, labels):
    test_split_proportion = config["test_split_proportion"]
    test_pivot = int(test_split_proportion * len(data))
    best_predictor, best_config = generate_predictor_from_random_config(config)
    best_predictor.train_on_data(
        config["batch_size"], config["epochs"], data[test_pivot:], labels[test_pivot:]
    )
    best_test_error = (
        best_predictor.get_loss(data[:test_pivot], labels[:test_pivot]) / test_pivot
    )
    progress_range = trange(config["model_count"])
    for i in progress_range:
        new_predictor, new_config = generate_predictor_from_random_config(config)
        new_predictor.train_on_data(
            config["batch_size"],
            config["epochs"],
            data[test_pivot:],
            labels[test_pivot:],
        )
        new_test_error = (
            new_predictor.get_loss(data[:test_pivot], labels[:test_pivot]) / test_pivot
        )
        if best_test_error > new_test_error:
            best_predictor, best_config = new_predictor, new_config
            best_test_error = new_test_error
        progress_range.set_description(
            "Trained Model {} with Loss {:.5f}. Current Optimal Loss: {:.5f}".format(
                i, new_test_error, best_test_error
            )
        )
        progress_range.refresh()
    return best_predictor, best_test_error, best_config


def create_data_from_data_stores(path, sensor_enabled):
    """
    Imports multiple episode data from the pickled episode data objects. 
    In order to do sample batches for the value function easily,
    creates a shifted states array (all states, but rotated by 1 for each episode)
    and is_non_terminal array (1s everywhere except last episode)
    """
    file_names = os.listdir(path)
    rewards, states, action, shifted_states, is_non_terminal = [], [], [], [], []

    for file_name in file_names:
        with open(os.path.join(path, file_name), "rb") as input_file:
            data = pickle.load(input_file)
            if len(data.states) == 0:
                continue
            episode_state = [np.concatenate(state) for state in data.states[1:]]
            states.append(np.concatenate(data.states[0]))
            states.extend(episode_state)
            shifted_states.extend(episode_state)
            shifted_states.append(np.concatenate(data.states[0]))

            multiplier = np.ones(len(data.states))
            multiplier[-1] = 0
            is_non_terminal.extend(multiplier)

            rewards.extend(data.rewards)
            action.extend(data.actions)

    if sensor_enabled:
        states = np.delete(np.asarray(states), 18, axis=1)
        shifted_states = np.delete(np.asarray(shifted_states), 18, axis=1)
    else:
        states = np.delete(np.asarray(states), range(8, 19), axis=1)
        shifted_states = np.delete(np.asarray(shifted_states), range(8, 19), axis=1)

    return (
        np.asarray(rewards),
        states,
        np.asarray(action),
        shifted_states,
        np.asarray(is_non_terminal),
    )


def get_state_scaling(data):
    rewards, states, action, shifted_states, is_non_terminal = data
    return np.mean(states, axis=0), np.std(states, axis=0)


def get_action_scaling(data):
    rewards, states, action, shifted_states, is_non_terminal = data
    return np.mean(action, axis=0), np.std(action, axis=0)


def pretrain_predictor_as_value_func(
    predictor, discount, data, epochs, batch_size, test_pivot
):
    rewards, states, action, shifted_states, is_non_terminal = data
    for _ in range(epochs):
        batch_index = (
            np.random.choice(range(len(states) - test_pivot), batch_size) + test_pivot
        )
        next_step_value = np.multiply(
            is_non_terminal[batch_index],
            predictor.evaluate(shifted_states[batch_index]).flatten(),
        )
        td_label = rewards[batch_index] + discount * next_step_value
        predictor.train_batch(states[batch_index], td_label.reshape((batch_size, 1)))


def get_value_func_loss(predictor, discount, data, test_pivot):
    rewards, states, _action, shifted_states, is_non_terminal = data
    next_step_value = predictor.evaluate(shifted_states[:test_pivot]).flatten()
    td_labels = rewards[:test_pivot] + discount * np.multiply(
        is_non_terminal[:test_pivot], next_step_value
    )
    return (
        predictor.get_loss(states[:test_pivot], td_labels.reshape((test_pivot, 1)))
        / test_pivot
    )


def create_value_func_from_hyper_param_search(config, data):
    print("Running random search on value critic...")
    test_split_proportion = config["test_split_proportion"]
    test_pivot = int(test_split_proportion * len(data[0]))
    states_shift, states_scale = get_state_scaling(data)
    config["scale_config"] = {
        "input_scale": states_scale,
        "input_shift": states_shift,
        "output_shift": 0,
        "output_scale": 1,
    }
    best_predictor, best_config = generate_predictor_from_random_config(config)
    pretrain_predictor_as_value_func(
        best_predictor,
        config["discount"],
        data,
        config["epochs"],
        config["batch_size"],
        test_pivot,
    )

    best_test_error = get_value_func_loss(
        best_predictor, config["discount"], data, test_pivot
    )

    progress_range = trange(config["model_count"])

    for i in progress_range:
        new_predictor, new_config = generate_predictor_from_random_config(config)
        pretrain_predictor_as_value_func(
            new_predictor,
            config["discount"],
            data,
            config["epochs"],
            config["batch_size"],
            test_pivot,
        )
        new_test_error = get_value_func_loss(
            new_predictor, config["discount"], data, test_pivot
        )
        if best_test_error > new_test_error:
            best_predictor, best_config = new_predictor, new_config
            best_test_error = new_test_error
        progress_range.set_description(
            "Trained Model {} with Loss {:.5f}. Current Optimal Loss: {:.5f}".format(
                i, new_test_error, best_test_error
            )
        )
        progress_range.refresh()
    return best_predictor, best_test_error, best_config


def create_actor_func_from_hyper_param_search(config, data):
    print("Running random search on actor...")

    _rewards, states, action, _shifted_states, _is_non_terminal = data
    states_shift, states_scale = get_state_scaling(data)
    action_shift, action_scale = get_action_scaling(data)
    config["scale_config"] = {
        "input_scale": states_scale,
        "input_shift": states_shift,
        "output_scale": action_scale,
        "output_shift": action_shift,
    }
    return create_predictor_from_hyper_param_search(config, states, action)


if __name__ == "__main__":
    import yaml

    actor_config_path = "Environment/training_configs/actor_hyperparam_search_config.yaml"
    critic_config_path = "Environment/training_configs/critic_hyperparam_search_config.yaml"
    training_data_dir = "training_data/Fixed_Walk_With_Sensor"
    data = create_data_from_data_stores(training_data_dir)

    with open(actor_config_path) as file:
        actor_config = yaml.load(file, Loader=yaml.FullLoader)
    create_actor_func_from_hyper_param_search(actor_config, data)

    with open(critic_config_path) as file:
        critic_config = yaml.load(file, Loader=yaml.FullLoader)
    create_value_func_from_hyper_param_search(critic_config, data)
