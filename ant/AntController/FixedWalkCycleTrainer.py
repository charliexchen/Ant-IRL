#!/usr/bin/python
import os
import pickle
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from ServoController.WalktCycleConfigParser import WalkCycle

class HaikuPredictorTrainer:
    def __init__(self, predictor, loss, learning_rate, decay, rng, steps=1, name="Ant"):
        self.training_data = WalkCycle().get_training_data(steps)
        state, _label = next(input_dataset)
        net_t = hk.without_apply_rng(hk.transform(predictor))
        self.params = net_t.init(rng, state)

        def _loss(params, old_states, new_states, learning_rate):
            predicted_state = net_t.apply(params, old_states)
            return learning_rate * loss(predicted_state, new_states)

        def _sgd(params, update):
            return params + update

        self._jit_sgd = jax.jit(_sgd)
        self._jit_loss = jax.jit(_loss)
        self._jit_grad = jax.jit(jax.grad(_loss))
        self._jit_predict = jax.jit(net_t.apply)

        self.learning_rate = learning_rate
        self.decay = decay
        self.generations = 0
        self.name = name

    def evaluate(self, states):
        return self._jit_predict(self.params, states)

    def _train_batch(self, states, labels):
        learning_rate = -self.learning_rate / len(states)
        update = self._jit_grad(self.params, states, labels, learning_rate)
        self.params = jax.tree_multimap(self._jit_sgd, self.params, update)
        self.learning_rate = self.learning_rate * self.decay
        self.generations += 1

    def get_loss(self, states, labels):
        return self._jit_loss(self.params, states, labels, 1 / len(states))

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


def gen_mlp_from_config(mlp_config, argument):
    layers = []
    for layer in mlp_config["layers"]:
        if layer["type"] == "linear":
            layers.append(hk.Linear(layer["size"]))
        elif layer["type"] == "sigmoid":
            layers.append(jax.nn.sigmoid)
        elif layer["type"] == "relu":
            layers.append(jax.nn.relu)
    mlp = hk.Sequential(layers)
    return mlp(argument) * mlp_config["scale"] + mlp_config["shift"]


def generate_controller_from_config(config):
    predictor = partial(gen_mlp, config["mlp"])
    if config['loss'] == "squared_loss":
        loss = squared_loss
    else:
        raise ValueError("Valid loss function not provided")
    rng = jax.random.PRNGKey(config["rng_key"])
    return HaikuPredictorTrainer(predictor, loss, config["learning_rate"], config["decay"], rng,
                                 config["steps"], config["ant"])


def squared_loss(predictions, labels):
    return jnp.sum(jnp.square(predictions - labels))


def net(current_position):
    mlp = hk.Sequential([
        hk.Linear(2048), jax.nn.sigmoid,
        hk.Linear(8)
    ])
    return mlp(current_position)


rng = jax.random.PRNGKey(42)
input_dataset = WalkCycle().get_training_data(1)

ac = HaikuPredictorTrainer(net, squared_loss, 0.001, 0.999999, rng, 1)
print("start!")
c = 0
loss = 1
current_pos, next_pos = next(input_dataset)
l = []
try:
    while True:
        ac._train_batch(np.asarray([current_pos]), np.asarray([next_pos]))
        _current_pos, next_pos = next(input_dataset)
        current_pos = ac.evaluate(current_pos)
        label = next_pos-current_pos
        l.append(ac.get_loss(np.asarray([current_pos]), np.asarray([next_pos])))
        c+=1
        if len(l)>1000:
            l.pop(0)

        if c % 100 == 0:
            print(sum(l)/len(l))
except KeyboardInterrupt:
    ac.save_params()
    print('Saving Parameters')



#
# try:
#     while True:
#         # import pdb;pdb.set_trace()
#         images = []
#         labels = []
#
#         for _ in range(256):
#             c += 1
#             img, lab = next(input_dataset)
#             images.append(img)
#             labels.append(lab)
#         images = np.array(images)
#         labels = np.array(labels)
#
#         images = np.random.normal(images, 0.1)
#         ac._train_batch(images, labels)
#         if c % (256 * 10) == 0:
#             loss = ac.get_loss(images, labels)
#             print(ac.generations, loss)
#     ac.save_params()
# except KeyboardInterrupt:
#     "saving parameters..."
#     ac.save_params()
