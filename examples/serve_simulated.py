
###############################################################################
# File serve_simulated.py
# Hosts and serves simulated images.
# Perhaps this is just for completion. I can't see any short-term use of this
# function.
#
###############################################################################
from pyfpm.web import create_server
import pyfpm.local as local
from pyfpm.devices import LaserAim, Camera
from pyfpm.fpmmath import iter_positions

size = (480, 640)
pup_rad = 40
overlap = 0.5
rmax = 320

with open('pinhole.png', "rb") as imageFile:
    image = imageFile.read()
client = local.SimClient(image, size, pup_rad)

app.run(host='0.0.0.0')
