import os
import pickle
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from FixedWalkCycleTrainer import generate_controller_from_config


class ActorCriticTrainer():
    def __init__(self, **kwargs):
        self.params = kwargs
        self.actor_transform = hk.without_apply_rng(hk.transform(actor_mlp))
        self.actor_weights = net_t.init(rng, state)
        self.critic_transform = hk.without_apply_rng(hk.transform(actor_mlp))
        self.actor_weights = net_t.init(rng, state)




