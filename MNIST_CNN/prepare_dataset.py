import idx2numpy as idx
import numpy as np
from scipy import ndimage
import idx2numpy as idx

# download the dataset from http://yann.lecun.com/exdb/mnist/

scale_factor = 32 / 28

image_array = idx.convert_from_file("../../mnist_ML/mnist/train-images.idx3-ubyte")
upscaled_array = np.zeros((image_array.shape[0], 32, 32), dtype=np.uint8)

for i, img in enumerate(image_array):
    upscaled_img = ndimage.zoom(img, scale_factor, order=1)
    upscaled_array[i] = upscaled_img

upscaled_array = upscaled_array.reshape(-1, 32, 32, 1)
upscaled_array.tofile("mnist_train_X.bin")
y_train = idx.convert_from_file("../../mnist_ML/mnist/train-labels.idx1-ubyte")
y_train.tofile("mnist_train_y.bin")

image_array = idx.convert_from_file("../../mnist_ML/mnist/t10k-images.idx3-ubyte")
upscaled_array = np.zeros((image_array.shape[0], 32, 32), dtype=np.uint8)

for i, img in enumerate(image_array):
    upscaled_img = ndimage.zoom(img, scale_factor, order=1)
    upscaled_array[i] = upscaled_img

upscaled_array = upscaled_array.reshape(-1, 32, 32, 1)
upscaled_array.tofile("mnist_test_X.bin")
y_test = idx.convert_from_file("../../mnist_ML/mnist/t10k-labels.idx1-ubyte")
y_test.tofile("mnist_test_y.bin")

