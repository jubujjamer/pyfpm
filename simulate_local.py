
###############################################################################
# File serve_simulated.py
# Hosts and serves simulated images.
# Perhaps this is just for completion. I can't see any short-term use of this
# function.
###############################################################################
from io import BytesIO
import time

import matplotlib.pyplot as plt
import h5py
from PIL import Image
import numpy as np

import pyfpm.local as local
from pyfpm.fpmmath import iter_positions, recontruct
from pyfpm.data import save_metadata

from scipy import misc
from StringIO import StringIO


# Simulation parameters
out_file = './outputs/pinhole_simulated.png'
out_hf = 'pinhole_3.h5'
## Obs: pup_rad = nx*NA/n where n is the refraction index of the medium
pupil_radius = 20
ns  = 0.3 # Comlement of the overlap between sampling pupils
phi_max = 20

# Opens input image as if it was sampled at pupil_pos = (0,0) with high
# resolution details
with open('pinhole_square.png', "r") as imageFile:
    image = imageFile.read()
    image_size =np.shape( misc.imread(StringIO(image),'RGB'))
client = local.SimClient(image, image_size, pupil_radius, ns)
iterator_list = list(iter_positions(pupil_radius, ns, phi_max, image_size, "leds"))

# Acquiring simulated images
task = "reconstruct"
if task is "acquire":
    with h5py.File(out_hf, 'w') as hf:
        print("I will take %s images" % len(iterator_list) )
        save_metadata(hf, image_size, iterator_list, pupil_radius, ns, phi_max)
        for index, theta, phi, power in iterator_list:
            print(index, theta, phi, power)
            img = client.acquire(theta, phi, power)
            hf.create_dataset(str(index), data=img)

elif task is "reconstruct":
    print("I will reconstruct on %s images" % len(iterator_list) )
    start_time = time.time()
    rec = recontruct(out_hf, debug=True, ax=None)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.imshow(rec), plt.gray()
    plt.show()

elif task is "inspect":
    with h5py.File(out_hf, 'r') as hf:
        for i in hf:
            print i
        for index, theta, phi, power in iterator_list:
            print(index)
            im_array = hf[str(int(index))][:]
            print np.shape(im_array)
            ax = plt.gca() or plt
            ax.imshow(im_array, cmap=plt.get_cmap('gray'))
            ax.get_figure().canvas.draw()
            plt.show(block=False)
