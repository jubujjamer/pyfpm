###############################################################################
# File micro_local.py
# Runs the microscope locally and doesn't hosts a web. To test in situ.
#
###############################################################################
from pyfpm import local
from pyfpm.devices import LaserAim, Camera

controller_port = '/dev/ttyACM'
try:
    laseraim = LaserAim(port = controller_port+str(0), theta = 0, phi = 0, power = 0)
except:
    print(controller_port + " not available, testing with another.")
    laseraim = LaserAim(port = controller_port+str(1), theta = 0, phi = 0, power = 0)

cam = Camera(video_id = 0)
client = local.Client(cam, laseraim)

for theta, phi, power in iter_positions(40, .5):
    img = client.acquire(theta, phi, power)
