import yaml
import matplotlib.pyplot as plt
import numpy as np
import time

from pyfpm.coordinates import PlatformCoordinates, phi_rot, theta_rot
pc = PlatformCoordinates(theta=0, phi=0, shift=0, height=60)
#
# print(pc.calculate_spot_center())
fig, ax = plt.subplots()
centers_accum = pc.spot_image(5, 'g')
image = pc.spot_image(40, 'r') + centers_accum
im = ax.imshow(image, cmap="hot")
fig.show()
# plt.show()

for phi in range(-70, 70, 5):
    pc.phi = phi
    print(pc.phi)
    # print(pc.phi_to_center())
    centers_accum = centers_accum + pc.spot_image(5, 'g')
    image = pc.spot_image(40, 'r') + centers_accum
    im.set_data(image)
    fig.canvas.draw()
    time.sleep(.01)
plt.show()
