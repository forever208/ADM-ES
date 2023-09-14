import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def imgs_to_npz():
    npz = []

    for img in os.listdir("./celebahq256_imgs/train"):
        img_arr = cv2.imread("./celebahq256_imgs/train/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # cv2默认为 bgr 顺序
        npz.append(img_arr)

    npz = np.array(npz)
    np.savez('celebaHQ256_train.npz', npz)
    print(f"{npz.shape} size array saved into celebaHQ256_train.npz")  # (27000, 256, 256, 3)


def show_images():
    x = np.load('./celebaHQ256_train.npz')['arr_0']
    plt.figure(figsize=(10, 10))
    for i in range(16):
        img = x[i, :, :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    # plt.savefig('./imgnet32_samples_4.jpg')
    plt.show()


if __name__ == '__main__':
    imgs_to_npz()
    # show_images()