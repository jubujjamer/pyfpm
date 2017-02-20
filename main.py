import yaml
import matplotlib.pyplot as plt
import numpy as np
import time

from pyfpm.coordinates import PlatformCoordinates

pc = PlatformCoordinates(theta=45, phi=60, shift=0, height=60)

print(pc.calculate_spot_center())

fig, ax = plt.subplots()
image = pc.spot_image()
im = ax.imshow(image, cmap="hot")
fig.show()
for theta in range(0, 360, 5):
    pc.theta = theta
    image = pc.spot_image()
    im.set_data(image)
    fig.canvas.draw()
    time.sleep(.01)
