from pathlib import Path
import random

import addict
import numpy as np
import torch
import yaml


def read_config(path: str) -> addict.Dict:
    with Path(path).open() as yfs:
        return addict.Dict(yaml.safe_load(yfs))


def reseed(new_seed: int) -> None:
    np.random.default_rng().bit_generator.state["state"]["state"] = new_seed
    random.SystemRandom().seed(int(new_seed))
    torch.manual_seed(min(new_seed, 0xffff_ffff_ffff_ffff))
