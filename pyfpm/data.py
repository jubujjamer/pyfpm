import os
import csv
import json
import yaml
import collections
import datetime
import numpy as np

HOME_FOLDER = os.path.expanduser("~/pyfpm")
CONFIG_FILE = os.path.join(HOME_FOLDER, "etc/config.yaml")
OUT_SAMLPING = os.path.join(HOME_FOLDER, "out_simulation")
OUT_SAMLPING = os.path.join(HOME_FOLDER, "out_sampling")

print(HOME_FOLDER, CONFIG_FILE)

def save_yaml_metadata(outname, cfg):
    base = os.path.splitext(outname)[0]
    outname = base + '.yaml'
    out_dict = cfg._asdict()
    timestamp = '{:%Y-%m-%d %H%M%S}'.format(datetime.datetime.now())
    out_dict['timestamp'] = timestamp
    with open(outname, 'w') as outfile:
        yaml.dump(out_dict, outfile, default_flow_style=False)
    return


def load_config():
    config_dict = yaml.load(open(CONFIG_FILE, 'r'))
    config = collections.namedtuple('config', config_dict.keys())
    cfg = config(*config_dict.values())
    return cfg


def generate_out_file(out_folder):
    """ File name with the date and hour to have one different file name
    to each measurment
    """
    out_file = os.path.join(out_folder,
                    '{:%Y-%m-%d_%H%M%S}'.format(datetime.datetime.now()))
    return out_file


def iter_dict(image_dict):
    for ((theta, phi), (img, power)) in image_dict.items():
        yield theta, phi, power, img


def open_sampled(filename):
    datafile = os.path.join(OUT_SAMLPING, filename)
    configfile = os.path.splitext(datafile)[0]+'.yaml'
    print(configfile)
    config_dict = yaml.load(open(configfile, 'r'))
    config = collections.namedtuple('config', config_dict.keys())
    file_cfg = config(*config_dict.values())
    return np.load(datafile)[()], file_cfg
