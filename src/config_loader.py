from pathlib import path

import yaml

class ConfigLoader:

    def __init__(self, config_path: str):
        config_path = path(config_path)
        config_path = config_path.resolve()
        config_path = str(config_path)

        try:
            with open(config_path, 'r') as config_file:
                self.data = yaml.safe_load(config_file)
        except FileNotFoundError:
            print(f"{config_file} not found")
        except yaml.YAMLError:
            print(f"Error loading YAML file: {yaml.YAMLError}")


    def parameters(self,):
        pass