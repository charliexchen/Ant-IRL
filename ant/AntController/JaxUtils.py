import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit


def squared_loss(predictions, labels):
    return jnp.sum(jnp.square(predictions - labels))


def normal_density(means, std, values):
    exponent = -jnp.divide(jnp.square(means - values), 2 * jnp.square(std))
    multiplier = 1 / jnp.sqrt(2 * jnp.pi) * std
    return multiplier * jnp.exp(exponent)


@jit
def sgd(learning_rate, params, update):
    return params + update * learning_rate


def gen_mlp_from_config(mlp_config, argument):
    layers = []
    for layer in mlp_config["layers"]:
        if layer["type"] == "linear":
            layers.append(hk.Linear(layer["size"]))
        elif layer["type"] == "sigmoid":
            layers.append(jax.nn.sigmoid)
        elif layer["type"] == "relu":
            layers.append(jax.nn.relu)
        elif layer["type"] == "lrelu":
            layers.append(jax.nn.leaky_relu)
        elif layer["type"] == "softmax":
            layers.append(jax.nn.softmax)
    mlp = hk.Sequential(layers)
    return mlp(argument) * mlp_config["scale"] + mlp_config["shift"]
