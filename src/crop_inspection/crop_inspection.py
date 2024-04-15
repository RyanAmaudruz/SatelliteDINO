



brightness_factor=2

####################################

# To run
import numpy as np
import matplotlib.pyplot as plt

img_0 = crops[0]
img_0_np = np.flip((img_0.to('cpu').numpy() * 255)[1:4, :, :], 0)

img_0_np_bright = (img_0_np * brightness_factor).clip(0, 255).astype(int)

plt.imshow(np.transpose(img_0_np_bright, (1, 2, 0)))
plt.show()
