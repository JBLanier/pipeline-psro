import argparse
import logging

import yaml

from mprl.utility_services.manager.manager import ManagerServer
from mprl.utility_services.utils import pretty_print

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to launch config YAML file", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    logging.basicConfig(level=logging.DEBUG)

    logger.info(f"launching with config:\n{pretty_print(config)}")

    manager = ManagerServer(config=config)
    manager.run()
