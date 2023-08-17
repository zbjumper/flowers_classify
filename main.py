import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# import torch
import torchvision.transforms as transforms

myimg = mpimg.imread('train_dataset/000/image_08000.jpg')
plt.imshow(myimg)
plt.show()
print(type(myimg), myimg.shape)


pil2tensor = transforms.ToTensor()
rgb_image = pil2tensor(myimg)

print(rgb_image[0][0]) # 一行像素的数据，归一化的
print(rgb_image.shape)

transforms.ToPILImage()(rgb_image[0]).show()
transforms.ToPILImage()(rgb_image[1]).show()
transforms.ToPILImage()(rgb_image[2]).show()