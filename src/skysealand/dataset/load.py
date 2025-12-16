import pathlib
from typing import TypedDict

import yaml


class DatasetSpec(TypedDict):
    train: pathlib.Path
    val: pathlib.Path
    test: pathlib.Path
    num_classes: int


def load_dataset_config(config_path: pathlib.Path) -> DatasetSpec:
    """
    Anchors the dataset directories for each split to their yaml config file location.

    Args:
        config_path: The path to the config summary yaml file for the dataset.
            This should be included with the dataset download.
            Defaults to ``data/data.yaml``.
            This directory must contain a ``train``, ``val``, *and* ``test`` directory.

    Returns:
        A dictionary of the train, val, and test dataset locations
        as well as metadata like the number of classes.
    """
    config_path = config_path.expanduser()
    base_dir = config_path

    with config_path.open() as f:
        config = yaml.safe_load(f)

    for split in ("train", "val", "test"):
        p = pathlib.Path(config[split])
        if not p.is_absolute():
            config[split] = (base_dir / p).resolve()
        else:
            config[split] = p

    config["num_classes"] = config.pop("nc")
    return config
