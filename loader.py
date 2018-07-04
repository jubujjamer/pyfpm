import matplotlib.pyplot as plt
import numpy as np

sample = np.load('sample_10.npy')
im_out = np.load('im_out_10.npy')

fis, axes = plt.subplots(nrows=1, ncols=3)
axes[0].imshow(sample)
axes[1].imshow(np.abs(im_out))
axes[2].imshow(np.angle(im_out))
plt.show()
