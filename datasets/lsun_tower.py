"""
Convert an LSUN lmdb database into a directory of images.
"""

import argparse
import io
import os

from PIL import Image
import lmdb
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_images(lmdb_path, image_size):
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=100, readonly=True)
    with env.begin(write=False) as transaction:
        cursor = transaction.cursor()
        for _, webp_data in cursor:
            img = Image.open(io.BytesIO(webp_data))
            width, height = img.size
            scale = image_size / min(width, height)
            img = img.resize(
                (int(round(scale * width)), int(round(scale * height))),
                resample=Image.BOX,
            )
            arr = np.array(img)
            h, w, _ = arr.shape
            h_off = (h - image_size) // 2
            w_off = (w - image_size) // 2
            arr = arr[h_off : h_off + image_size, w_off : w_off + image_size]
            yield arr


def dump_images(out_dir, images, prefix):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"{prefix}_{i:07d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", help="new image size", type=int, default=256)
    parser.add_argument("--prefix", help="class name", type=str, default="bedroom")
    parser.add_argument("lmdb_path", help="path to an LSUN lmdb database")
    parser.add_argument("out_dir", help="path to output directory")
    args = parser.parse_args()

    images = read_images(args.lmdb_path, args.image_size)
    dump_images(args.out_dir, images, args.prefix)


def imgs_to_npz():
    npz = []

    for img in os.listdir("./lsun_tower_train"):
        img_arr = cv2.imread("./lsun_tower_train/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # cv2默认为 bgr 顺序
        npz.append(img_arr)

    output_npz = np.array(npz)
    np.savez('lsun_tower_train.npz', output_npz)
    print(f"{output_npz.shape} size array saved into lsun_tower_train.npz")  # (708264, 64, 64, 3)


def show_images():
    x = np.load('./lsun_tower_train.npz')['arr_0']
    plt.figure(figsize=(10, 10))
    for i in range(16):
        img = x[i, :, :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    # plt.savefig('./imgnet32_samples_4.jpg')
    plt.show()


def generate_inference_batch():
    x = np.load('./lsun_tower_train.npz')['arr_0']
    inference_batch = []
    random_ind = np.random.choice(range(x.shape[0]), size=50000, replace=False).tolist()
    print(len(random_ind))

    for i in random_ind:
        inference_batch.append(x[i, :, :, :])

    output_npz = np.array(inference_batch)
    np.savez('lsun_tower_train_50k.npz', output_npz)
    print(f"{output_npz.shape} size array saved into lsun_tower_train_50k.npz")  # (50000, 64, 64, 3)


if __name__ == "__main__":
    # main()
    # imgs_to_npz()
    # show_images()
    generate_inference_batch()
