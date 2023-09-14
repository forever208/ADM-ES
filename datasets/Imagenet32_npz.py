import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def ImageNet_subset():
    output = []

    for i in range(1, 11):
        filename = 'train_data_batch_' + str(i) + '.npz'
        raw_data = np.load(filename)
        print(f"{filename} loaded")
        labels = list(raw_data['labels'])
        print(f"{raw_data['data'].shape[0]} images founded")

        x = raw_data['data']
        x = np.dstack((x[:, :1024], x[:, 1024:2 * 1024], x[:, 2 * 1024:]))
        x = x.reshape((x.shape[0], 32, 32, 3))

        for ind, label in enumerate(labels):
            if label in [k for k in range(1, 101)]:
                output.append(x[ind])

        print(f"total {len(output)} images with label in (1, 100) founded")
        print('')

    output_array = np.array(output)
    np.savez('ImageNet32_1_to_100.npz', output_array)
    print(f"{output_array.shape} size array saved into ImageNet32_1_to_100.npz")


def merge_all_train_dataset():
    output = None
    for i in range(1, 11):
        filename = './Imagenet32_train_npz/train_data_batch_' + str(i) + '.npz'
        raw_data = np.load(filename)
        print(f"{filename} loaded")
        labels = list(raw_data['labels'])
        print(f"{raw_data['data'].shape[0]} images founded")

        # reshape data into 4D array (num_images, 32, 32, 3)
        x = raw_data['data']
        x = np.dstack((x[:, :1024], x[:, 1024:2 * 1024], x[:, 2 * 1024:]))
        x = x.reshape((x.shape[0], 32, 32, 3))

        if output is None:
            output = x
        elif output.any():
            output = np.concatenate((output, x), axis=0)

    output_array = np.array(output)
    np.savez('ImageNet32_train_all.npz', output_array)
    print(f"{output_array.shape} size array saved into ImageNet32_train_all.npz")


def show_images():
    x = np.load('ImageNet32_train_all.npz')['arr_0']
    plt.figure(figsize=(10, 5))
    for i in range(32):
        img = x[i, :, :, :]
        plt.subplot(4, 8, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # ImageNet_subset()
    # merge_all_train_dataset()
    show_images()