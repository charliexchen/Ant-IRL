#!/usr/bin/python
import haiku as hk
import jax.numpy as jnp
import jax
from WalktCycleConfigParser import WalkCycle
import pickle


class AntController:
    def __init__(self, net, loss, learning_rate, decay, rnd_key=42):
        rng = jax.random.PRNGKey(key)
        self.params = hk.transform(net, rng)
        self.grad = jax.jit()
        self._loss = loss


def squared_loss(prediction, label):
    return jnp.dot(prediction - label, prediction - label)


def net(current_position):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(8)
    ])
    return (mlp(current_position) * 2) - 1


def loss_fn(current_position, label):
    predicted_position = net(current_position)
    return squared_loss(predicted_position, label)


loss_fn_t = hk.transform(loss_fn)

rng = jax.random.PRNGKey(42)
wc = WalkCycle()
input_dataset = wc.get_training_data()
current_position, label = next(input_dataset)

net_t = hk.transform(net)
net_t.init(rng, current_position)

params = loss_fn_t.init(rng, current_position, label)

print(jax.jit(net_t.apply)(params, rng, jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])))


def sgd(param, update):
    return param - 0.0001 * update


print("start!")
c = 0
j_grads = jax.jit(jax.grad(loss_fn_t.apply))
for images, labels in input_dataset:
    c += 1
    grads = j_grads(params, None, images, labels)
    params = jax.tree_multimap(sgd, params, grads)
    if c % 100 == 0:
        loss = loss_fn_t.apply(params, None, current_position, label)
        print("Generation " + str(c) + ", Loss: " + str(loss))
    if c % 10000 == 0:
        break
pickle.dump(params, open("params.p", "wb"))
