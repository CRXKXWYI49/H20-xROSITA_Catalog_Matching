from pathlib import Path
import yaml

class ConfigLoader:

    def __init__(self, config_path: str):
        config_path: Path = Path(config_path).resolve()

        try:
            config_file = open(str(config_path), 'r')
            self.yaml_dict = yaml.safe_load(config_file)
            self.load_member_vars()

        except FileNotFoundError:
            print(f"{config_file} not found")

        except yaml.YAMLError:
            print(f"Error loading YAML file: {yaml.YAMLError}")


    def load_member_vars(self):
        data = self.yaml_dict
        
        self.EROSITA_PATH = data['eROSITA_data_path']
        self.H20_PATH = data['H20_data_path']
