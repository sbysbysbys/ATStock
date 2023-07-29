import matplotlib.pyplot as plt
import yaml

config_path = "vis.yaml"

def vis_daily():
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    daily_dir = config["daily"]["dir"]
    print(daily_dir)

if __name__ == "__main__":
    print("visulization/utils.py")
    vis_daily()