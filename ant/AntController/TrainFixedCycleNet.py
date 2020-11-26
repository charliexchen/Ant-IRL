#!/usr/bin/python
import haiku as hk
import jax.numpy as jnp
import jax
from jax import jit
import numpy as np
from ServoController.WalktCycleConfigParser import WalkCycle
import pickle
import os
from functools import partial


class AntController:
    def __init__(self, net, loss, learning_rate, decay, rng, steps=3, name="Ant"):
        self.training_data = WalkCycle().get_training_data(steps)
        state, _label = next(input_dataset)

        net_t = hk.without_apply_rng(hk.transform(net))
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
            if not os.path.exists(f"NetConfigs/{self.name}"):
                os.makedirs(f"NetConfigs/{self.name}")
            file_name = f"NetConfigs/{self.name}/params_{self.generations}_gen.p"
        pickle.dump(self.params, open(file_name, "wb"))

    def get_params(self, file_name=None):
        if file_name is None:
            max_gen = -1
            for file_name_ in os.listdir(f"NetConfigs/{self.name}/"):
                generation = int("".join([s for s in file_name_ if s.isdigit()]))
                if generation > max_gen:
                    file_name = file_name_
        self.params = pickle.load(open(file_name, "rb"))


def squared_loss(predictions, labels):
    return jnp.sum(jnp.square(predictions - labels))


def net(current_position):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.sigmoid,
        hk.Linear(256), jax.nn.sigmoid,
        hk.Linear(8)
    ])
    return mlp(current_position)


input_dataset = WalkCycle().get_training_data(1)

rng = jax.random.PRNGKey(42)
ac = AntController(net, squared_loss, 0.01, 0.999, rng, 1)
print("start!")
c = 0
loss = 1

try:
    while True:
        # import pdb;pdb.set_trace()
        images = []
        labels = []

        for _ in range(256):
            c += 1
            img, lab = next(input_dataset)
            images.append(img)
            labels.append(lab)
        images = np.array(images)
        labels = np.array(labels)

        images = np.random.normal(images, 0.1)
        ac._train_batch(images, labels)
        if c %  (256*10 )== 0:
            loss = ac.get_loss(images, labels)
            print(ac.generations, loss)
    ac.save_params()
except KeyboardInterrupt:
    "saving parameters..."
    ac.save_params()
