import os
import csv
import json
import yaml
import collections
import datetime
import numpy as np

try:
    os.environ["SUDO_USER"]
    HOME_FOLDER = os.path.join("/home/pi/pyfpm")
except:
    HOME_FOLDER = os.path.expanduser("~/git/pyfpm")
ETC_FOLDER = os.path.join(HOME_FOLDER, "etc")
CONFIG_FILE = os.path.join(HOME_FOLDER, "etc/config.yaml")
OUT_SIMULATION = os.path.join(HOME_FOLDER, "out_simulation")
OUT_SAMLPING = os.path.join(HOME_FOLDER, "out_sampling")

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


def load_model_file(model_name):
    model_file = os.path.join(ETC_FOLDER, model_name)
    model_dict = yaml.load(open(model_file, 'r'))
    model = collections.namedtuple('config', model_dict.keys())
    model_cfg = model(*model_dict.values())
    return model_cfg

def save_model(model_name, model):
    model_file = os.path.join(ETC_FOLDER, model_name)
    with open(model_file, 'w') as outfile:
        yaml.dump(model, outfile, default_flow_style=False)
    return

def generate_out_file(out_folder=OUT_SIMULATION, fname=None):
    """ File name with the date and hour to have one different file name
    to each measurment
    """
    outdir = os.path.join(HOME_FOLDER, out_folder)
    if fname is None:
        out_file = os.path.join(outdir,
                        '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now()))
    else:
        out_file = os.path.join(outdir, fname)
    return out_file


def iter_dict(image_dict):
    for ((theta, phi), (img, power)) in image_dict.items():
        yield theta, phi, power, img

def open_sampled(filename, mode='sampling'):
    if mode == 'sampling':
        datafile = os.path.join(OUT_SAMLPING, filename)
    if mode == 'simulation':
        datafile = os.path.join(OUT_SIMULATION, filename)
    configfile = os.path.splitext(datafile)[0]+'.yaml'
    config_dict = yaml.load(open(configfile, 'r'))
    config = collections.namedtuple('config', config_dict.keys())
    file_cfg = config(*config_dict.values())
    return np.load(datafile, encoding='bytes')[()], file_cfg
