import yaml
import matplotlib.pyplot as plt
import numpy as np

from pyfpm.coordinates import PlatformCoordinates

pc = PlatformCoordinates(theta=45, phi=45, shift=0, height=60)

print(pc.expected_spot_center())
