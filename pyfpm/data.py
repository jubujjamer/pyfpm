import os
import csv
import json

def save_to_folder(folder, image_dict):
    with open('meta.csv', 'wb') as csvfile:
        cw = csv.writer(csvfile, delimiter=',')
        cw.writerow(('theta', 'phi', 'power', 'filename'))
        for ndx, ((theta, phi), (img, power)) in enumerate(image_dict.items()):
            filename = 'file%05d.png' % ndx
            with open(os.path.join(folder, filename), 'wb') as fp:
                fp.write(img)
            cw.writerow((theta, phi, power, filename))

def load_from_folder(folder):
    out = dict()
    with open(os.path,join(folder, 'meta.csv'), 'rb') as csvfile:
        csv.readline()
        cr = csv.reader(csvfile, delimiter=',')
        for theta, phi, power, filename in reader:
            with open(os.path,join(folder, filename)) as fi:
                out[(theta, phi)] = fi.read(), power
    return out

def iter_dict(image_dict):
    for ((theta, phi), (img, power)) in image_dict.items():
        yield theta, phi, power, img


def json_savemeta(json_file, image_size, pupil_radius, theta_max,
                  theta_step, phi_max):
    data = {'image_size': image_size, 'pupil_radius': pupil_radius,
            'theta_max': theta_max, 'theta_step': theta_step,
            'phi_max': phi_max}
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile,
                  indent=4, sort_keys=True, separators=(',', ':'))


def json_loadmeta(json_file):
    with open(json_file) as json_data:
        return json.load(json_data)


def save_metadata(hf, image_size, iterator_list, pupil_radius, ns, phi_max):
    hf.create_dataset("image_size",     data = image_size   )
    hf.create_dataset("iterator_list",  data = iterator_list) # Double check
    hf.create_dataset("pupil_radius",   data = pupil_radius )
    hf.create_dataset("ns",             data = ns           )
    hf.create_dataset("phi_max",        data = phi_max      )

def get_metadata(hf):
    image_size = hf["image_size"][...]
    iterator_list = hf["iterator_list"][...]
    pupil_radius = hf["pupil_radius"][...]
    ns = hf["ns"][...]
    phi_max = hf["phi_max"][...]
    print("get_metadata", image_size[:], iterator_list[:], pupil_radius[...], ns, phi_max)
    return  image_size, iterator_list, pupil_radius, ns, phi_max
