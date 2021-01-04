#!/usr/bin/python
import functools
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pickle
from jax import jit
from jax.tree_util import partial

from AntController.JaxUtils import squared_loss, gen_mlp_from_config
from ServoController.WalktCycleConfigParser import WalkCycle


class HaikuPredictor:
    """
    Haiku and Jax are amazing, but since they work off pure functions, this object helps storing, exporting and
    importing the params
    """

    def __init__(
            self, predictor, input_shape, learning_rate, rng, loss_func=None, name="Ant"
    ):
        self.net_t = hk.without_apply_rng(hk.transform(predictor))
        self.loss_func = loss_func
        self.params = self.net_t.init(rng, np.ones(input_shape))
        self.learning_rate = learning_rate
        self.generations = 0
        self.name = name
        self.optimizer = optax.adam(learning_rate, b1=0.5, b2=0.9)
        self.optimizer_state = self.optimizer.init(self.params)

        self.input_scale, self.input_shift = (1, 0)
        self.output_scale, self.output_shift = (1, 0)

    def set_scaling(self, input_scale, input_shift, output_scale, output_shift):
        self.input_scale, self.input_shift = input_scale, input_shift
        self.output_scale, self.output_shift = output_scale, output_shift

    def _normalise_input(self, inputs):
        return jnp.divide(inputs - self.input_shift, self.input_scale)

    def _normalise_output(self, outputs):
        return jnp.divide(outputs - self.output_shift, self.output_scale)

    def _scale_input(self, norm_inputs):
        return jnp.multiply(norm_inputs, self.input_scale) + self.input_shift

    def _scale_output(self, norm_outputs):
        return jnp.multiply(norm_outputs, self.output_scale) + self.output_shift

    @functools.partial(jax.jit, static_argnums=0)
    def _loss(self, params, data, labels):
        normalised_inputs = self._normalise_input(data)
        normalised_labels = self._normalise_output(labels)
        predictions = self.net_t.apply(params, normalised_inputs)
        return self.loss_func(predictions, normalised_labels)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate(self, params, data):
        normalised_inputs = self._normalise_input(data)
        normalised_outputs = self.net_t.apply(params, normalised_inputs)
        return self._scale_output(normalised_outputs)

    @functools.partial(jax.jit, static_argnums=0)
    def _train_batch(self, params, states, labels, optimizer_state):
        value, gradient = jax.value_and_grad(self._loss)(params, states, labels)
        update, new_optimizer_state = self.optimizer.update(gradient, optimizer_state)
        return optax.apply_updates(params, update), new_optimizer_state, value

    def get_loss(self, states, labels):
        return self._loss(self.params, states, labels)

    def evaluate(self, covariates):
        return self._evaluate(self.params, covariates)

    def train_batch(self, states, labels):
        self.params, self.optimizer_state, batch_loss = self._train_batch(
            self.params, states, labels, self.optimizer_state
        )
        self.generations += 1
        return batch_loss

    def save_params(self, file_name=None):
        if file_name is None:
            if not os.path.exists(f"configs/{self.name}"):
                os.makedirs(f"configs/{self.name}")
            file_name = f"configs/{self.name}/params_{self.generations}_gen.p"
        pickle.dump(self.params, open(file_name, "wb"))

    def get_params(self, file_name=None):
        if file_name is None:
            # gets the latest snapshot if not specified
            files_names = os.listdir(f"configs/{self.name}/")
            file_name = max(files_names, key=os.path.getctime)
        self.params = pickle.load(open(file_name, "rb"))

    @staticmethod
    def generate_controller_from_config(config):
        if "path" in config:
            predictor = HaikuPredictor.get_model_from_saved_file(config["path"])
            predictor.optimizer = optax.adam(
                config["new_learning_rate"], b1=0.5, b2=0.9
            )
            return predictor
        predictor = partial(gen_mlp_from_config, config["mlp"])
        if config["loss"] == "squared_loss":
            loss = squared_loss
        else:
            raise ValueError("Valid loss function not provided")
        rng = jax.random.PRNGKey(config["rng_key"])
        predictor = HaikuPredictor(
            predictor,
            config["input_shape"],
            config["learning_rate"],
            rng,
            loss,
            config["name"],
        )
        if "scale_config" in config:
            scale_config = config["scale_config"]
            predictor.set_scaling(
                np.asarray(scale_config["input_scale"]),
                np.asarray(scale_config["input_shift"]),
                np.asarray(scale_config["output_scale"]),
                np.asarray(scale_config["output_shift"]),
            )
        return predictor

    def train_on_data(self, batch_size, epochs, data, labels):
        losses = []
        for _ in range(epochs):
            batch_index = np.random.choice(range(len(data)), batch_size)
            loss = self.train_batch(data[batch_index], labels[batch_index])
            losses.append(loss)
        return losses

    @staticmethod
    def get_model_from_saved_file(path):
        config, params, optimizer_state = pickle.load(open(path, "rb"))
        predictor = HaikuPredictor.generate_controller_from_config(config)
        predictor.params = params
        predictor.optimizer_state = optimizer_state
        return predictor


if __name__ == "__main__":
    """
    Quick loop to import a a config from a yaml and try to train it on a fixed walk loop to verify testing.
    """
    import yaml

    path = "AntController/configs/fixed_trainer_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    ant_controller = HaikuPredictor.generate_controller_from_config(config)
    input_dataset = WalkCycle().get_training_data(1)
    current_pos, next_pos = next(input_dataset)
    print("start!")
    c = 0
    l = []
    try:
        while True:
            ant_controller.train_batch(
                np.asarray([current_pos]), np.asarray([next_pos])
            )
            _current_pos, next_pos = next(input_dataset)
            current_pos = ant_controller.evaluate(current_pos)
            label = next_pos - current_pos
            loss = ant_controller.get_loss(
                np.asarray([current_pos]), np.asarray([next_pos])
            )
            l.append(loss)
            c += 1
            if len(l) > 1000:
                l.pop(0)
            if c % 100 == 0:
                print(
                    str.format(
                        "current loss: {0:.5f}, running average: {0:.5f}",
                        loss,
                        sum(l) / len(l),
                    )
                )
    except KeyboardInterrupt:
        ant_controller.save_params()
        print("Saving Parameters")
