from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(
    path: Path = DEFAULT_CONFIG,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)