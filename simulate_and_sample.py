###############################################################################
# File simulate_and_sample.py
# With an already hosted server aqcuires the microscope image from the remote
# client and simulates the same image from the local computer.
#
###############################################################################
from pyfpm import web

client = web.Client(url)

for theta, phi, power in iter_positions(40, .5):
    img = client.acquire(theta, phi, power)

# procesas
