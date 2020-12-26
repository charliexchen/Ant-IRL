#!/usr/bin/python
import os
import pickle
from jax.tree_util import partial
import functools
import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

from ServoController.WalktCycleConfigParser import WalkCycle
from AntController.JaxUtils import squared_loss, gen_mlp_from_config


@jit
def _sgd(learning_rate, params, update):
    return params + update * learning_rate


class HaikuPredictor:
    """Haiku and Jax are amazing, but since they work off pure functions, this object helps storing, exporting and
    importing the params"""
    def __init__(self, predictor, input_shape, learning_rate, decay, rng, loss_func=None, name="Ant"):
        self.net_t = hk.without_apply_rng(hk.transform(predictor))
        self.loss_func = loss_func
        self.params = self.net_t.init(rng, np.ones(input_shape))
        self._jit_predict = jax.jit(self.net_t.apply)
        self.learning_rate = learning_rate
        self.decay = decay
        self.generations = 0
        self.name = name

    @functools.partial(jax.jit, static_argnums=0)
    def _loss(self, params, covariates, labels):
        predictions = self.net_t.apply(params, covariates)
        return self.loss_func(predictions, labels)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate(self, params, covariates):
        return self.net_t.apply(params, covariates)

    def evaluate(self, covariates):
        return self._evaluate(self.params, covariates)

    @functools.partial(jax.jit, static_argnums=0)
    def _train_batch(self, params, states, labels, learning_rate):
        update = jax.grad(self._loss)(params, states, labels)
        return jax.tree_multimap(jax.tree_util.partial(_sgd, learning_rate), params, update)

    def train_batch(self, states, labels):
        learning_rate = -self.learning_rate / len(states)
        self.params = self._train_batch(self.params, states, labels, learning_rate)
        self.learning_rate = self.learning_rate * self.decay
        self.generations += 1

    def get_loss(self, states, labels):
        return self._loss(self.params, states, labels)

    def save_params(self, file_name=None):
        if file_name is None:
            if not os.path.exists(f"configs/{self.name}"):
                os.makedirs(f"configs/{self.name}")
            file_name = f"configs/{self.name}/params_{self.generations}_gen.p"
        pickle.dump(self.params, open(file_name, "wb"))

    def get_params(self, file_name=None):
        if file_name is None:
            max_gen = -1
            for file_name_ in os.listdir(f"configs/{self.name}/"):
                generation = int("".join([s for s in file_name_ if s.isdigit()]))
                if generation > max_gen:
                    file_name = file_name_
        self.params = pickle.load(open(file_name, "rb"))

    @staticmethod
    def generate_controller_from_config(config):
        predictor = partial(gen_mlp_from_config, config["mlp"])
        if config['loss'] == "squared_loss":
            loss = squared_loss
        else:
            raise ValueError("Valid loss function not provided")
        rng = jax.random.PRNGKey(config["rng_key"])
        return HaikuPredictor(predictor, config["input_shape"],
                              config["learning_rate"], config["decay"], rng,
                              loss, config["name"])

if __name__ == "__main__":
    import yaml

    path = "AntController/configs/fixed_trainer_config.yaml"
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    ant_controller = HaikuPredictor.generate_controller_from_config(config)
    input_dataset = WalkCycle().get_training_data(1)
    current_pos, next_pos = next(input_dataset)
    print("start!")
    c = 0
    loss = 1
    l = []
    try:
        while True:
            ant_controller.train_batch(np.asarray([current_pos]), np.asarray([next_pos]))
            _current_pos, next_pos = next(input_dataset)
            current_pos = ant_controller.evaluate(current_pos)
            label = next_pos - current_pos
            l.append(ant_controller.get_loss(np.asarray([current_pos]), np.asarray([next_pos])))
            c += 1
            if len(l) > 1000:
                l.pop(0)

            if c % 100 == 0:
                print(sum(l) / len(l))
    except KeyboardInterrupt:
        ant_controller.save_params()
        print('Saving Parameters')
