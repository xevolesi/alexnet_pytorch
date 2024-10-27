from pathlib import Path

import addict
import yaml


def read_config(path: str) -> addict.Dict:
    with Path(path).open() as yfs:
        return addict.Dict(yaml.safe_load(yfs))
