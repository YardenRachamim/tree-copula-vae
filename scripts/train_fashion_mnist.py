from pathlib import Path
import sys

import pyrallis

from tree_copula_vae.fashion_mnist.config import Config
from tree_copula_vae.fashion_mnist.train import run


# DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "fashion_mnist_presets" / "SN-TC_GC-R1GC.yml"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "fashion_mnist_presets" / "SN-TC_ST-R1GC.yml"


def main() -> None:
    config_path = DEFAULT_CONFIG_PATH
    if len(sys.argv) > 1 and sys.argv[1] == "--config_path":
        config_path = Path(sys.argv[2])
        del sys.argv[1:3]
    config = pyrallis.parse(config_class=Config, config_path=str(config_path))
    run(config, config_path)


if __name__ == "__main__":
    main()