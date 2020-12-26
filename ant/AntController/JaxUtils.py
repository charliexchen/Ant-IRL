import jax
import jax.numpy as jnp

import haiku as hk

def squared_loss(predictions, labels):
    return jnp.sum(jnp.square(predictions - labels))


def gen_mlp_from_config(mlp_config, argument):
    layers = []
    for layer in mlp_config["layers"]:
        if layer["type"] == "linear":
            layers.append(hk.Linear(layer["size"]))
        elif layer["type"] == "sigmoid":
            layers.append(jax.nn.sigmoid)
        elif layer["type"] == "relu":
            layers.append(jax.nn.relu)
        elif layer["type"] == "softmax":
            layers.append(jax.nn.softmax)
    mlp = hk.Sequential(layers)
    return mlp(argument) * mlp_config["scale"] + mlp_config["shift"]
