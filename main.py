import yaml
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools as it

from pyfpm.coordinates import PlatformCoordinates
import pyfpm.data as dt

CONFIG_FILE = 'config.yaml'
config_dict = yaml.load(open(CONFIG_FILE, 'r'))
cfg = dt.load_config(CONFIG_FILE)

pc = PlatformCoordinates(theta=0, phi=0, height=cfg.sample_height, cfg=cfg)

phi_min, phi_max, phi_step = cfg.phi
theta_min, theta_max, theta_step = cfg.theta
theta_range = range(theta_min, theta_max, theta_step)
# phi_range = range(phi_min, phi_max, phi_step)
phi_range = [20]

phi_corr_list = list()
theta_corr_list = list()
pc.platform_tilt = [0, 5]
pc.source_center = [0, 3] #[xcoord, ycoord] of the calibrated center
pc.source_tilt = [0, 0]
pc.height = 97
for t, p in it.product(theta_range, phi_range):
    pc.set_coordinates(theta=t, phi=p, units='degrees')
    t_corr, p_corr = pc.source_coordinates(mode='angular')
    phi_corr_list.append(p_corr)
    theta_corr_list.append(t_corr)
    print("theta: %i, phi: % i | theta_corr: %.2f phi_corr %.2f" % (t, p, t_corr, p_corr))
plt.plot(theta_corr_list, phi_corr_list)

# phi_corr_list = list()
# theta_corr_list = list()
# pc.platform_tilt = [45, 10]
# for t, p in it.product(theta_range, phi_range):
#     pc.set_coordinates(theta=t, phi=p, units='degrees')
#     t_corr, p_corr = pc.source_coordinates(mode='angular')
#     phi_corr_list.append(p_corr)
#     theta_corr_list.append(t_corr)
# plt.plot(theta_corr_list, phi_corr_list)

plt.show()

pc.set_coordinates(theta=270, phi=10, units='degrees')
pc.platform_tilt = [0, 0]
t_corr, p_corr = pc.source_coordinates(mode='angular')
print("theta: %i, phi: % i | theta_corr: %.2f phi_corr %.2f" % (270, 10, t_corr, p_corr))



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


# pc.generate_model(model='nomodel')
# for phi in range(75, 100):
#     pc.phi = phi
#     print(pc.parameters_to_platform())
