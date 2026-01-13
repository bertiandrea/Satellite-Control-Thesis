# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

SEED = 420
NUM_ENVS = 4096
EPISODE_LENGTH = 240.0
HEADLESS = False
DEBUG_ARROWS = True
LOG_TRAJECTORIES = True

DR_RANDOMIZATION = False
EXPLOSION = False

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": True,
    "seed": SEED,

    "profile": False,

    "headless": HEADLESS,

    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,

        "max_episode_length": EPISODE_LENGTH,

        "debug_arrows": DEBUG_ARROWS,
        "debug_prints": False,
        "discretize_starting_pos": True,
        "log_trajectories": LOG_TRAJECTORIES,

        "asset": {
            "assetRoot": str(Path(__file__).resolve().parent.parent),
            "assetFileName": "satellite.urdf",
            "assetName": "satellite",
        }
    },
    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            
            "learning_rate_scheduler" : KLAdaptiveRL, # NOT SERIALIZABLE
            "state_preprocessor" : RunningStandardScaler, # NOT SERIALIZABLE
            "value_preprocessor" : RunningStandardScaler, # NOT SERIALIZABLE
            "rewards_shaper" : lambda rewards, timestep, timesteps: rewards * 0.01, # NOT SERIALIZABLE
        },
        "trainer": {
            "timesteps": int(MAX_EPISODE_LENGTH / ( 1.0 / 60.0 )),
            "disable_progressbar": False,
            "headless": HEADLESS,
            "stochastic_evaluation": False,
        },
    },
    
    # --- reward -----------------------------------------------------------
    "reward": {
        "log_reward": True,
        "log_reward_interval": 100,  # steps
    },

    # --- explosion ---------------------------------------------------------
    "explosion": {
        "enabled": EXPLOSION,
        "explosion_time": 60,  # seconds
    },

    # --- dr_randomization -------------------------------------------------
    "dr_randomization": {
        "enabled": DR_RANDOMIZATION,
        "dr_params": {
            "observations": {
                "distribution": "uniform", # "uniform" or "gaussian"
                "operation": "scaling", # "scaling" or "addition"
                "range": [0.9, 1.1], # gaussian: [mu, var], uniform: [low, high]
                "quaternion_noise": 0.1,
            },
            "actions": {
                "distribution": "uniform", # "uniform" or "gaussian"
                "operation": "scaling", # "scaling" or "addition"
                "range": [0.9, 1.1], # gaussian: [mu, var], uniform: [low, high]
            },
            "actor_params": {
                "satellite": {
                    "color": True,
                    "rigid_body_properties": {
                        "inertia": {
                            "distribution": "uniform", # "uniform" or "gaussian"
                            "operation": "scaling", # "scaling" or "addition"
                            "range": [0.9, 1.1], # gaussian: [mu, var], uniform: [low, high]
                        }
                    }
                }
            }
        }
    }
}