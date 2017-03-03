import yaml
import matplotlib.pyplot as plt
import numpy as np
import time

from pyfpm.coordinates import PlatformCoordinates
pc = PlatformCoordinates(theta=0, phi=0, shift=100, height=60)
#
# print(pc.calculate_spot_center())
# fig, ax = plt.subplots()
# centers_accum = pc.spot_image(5, 'g')
# image = pc.spot_image(40, 'r') + centers_accum
# im = ax.imshow(image, cmap="hot")
# fig.show()
# # # plt.show()
# #
# for theta in range(0, 360, 5):
#     pc.theta = theta
#     print(pc.phi_to_center())
#     centers_accum = centers_accum + pc.spot_image(5, 'g')
#     image = pc.spot_image(40, 'r') + centers_accum
#     im.set_data(image)
#     fig.canvas.draw()
#     time.sleep(.01)
# plt.show()


pc.generate_model()
# for phi in range(60,81):
#     pc.phi = phi
#     print(pc.parameters_to_platform())
