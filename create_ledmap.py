import itertools as it
import pyfpm.data as dt

cfg = dt.load_config()

led_center = 15
led_disp = (int(cfg.array_size)+1)//2
led_range = range(led_center-led_disp, led_center+led_disp)
ledmap = it.product(led_range, led_range)

ss_dict = {}
for led in ledmap:
    ss_dict[(led[0], led[1])] = 5E5

print(ss_dict)
