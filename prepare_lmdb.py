import lmdb
import os
from preprocess_images import process_single_image
from config import TRAINING_IMAGE_RESOLUTION
import pickle
import cv2
import numpy as np
import msgpack
from tqdm import tqdm
import lz4.frame

def imname_to_target(name:str) -> tuple[float]:
    """Parses image names of format x{x_value}_y{y_value}.jpg"""
    name = name.split('.jpg')[0]
    x, y = name.split("_")
    x = float(x[1:])
    y = float(y[1:])
    return x, y

def create_lmdb_from_images(
    image_dir, lmdb_path, start_index=0, stop_index=None, size=None, resolution=(512, 512), use_compression=True
):
    # Get filenames
    image_names = [
        str(f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
    ]

    # Open lmdb
    if size is None:
        r1, r2 = resolution
        size = r1 * r2 * 8 * len(image_names)
    env = lmdb.open(lmdb_path, map_size=size)

    if stop_index is None:
        stop_index = len(image_names)

    # keys_file
    keys_file = open(os.path.join(lmdb_path, "keys.txt"), "+w")

    # Store images
    with env.begin(write=True) as txn:
        try:
            pbar = tqdm(range(start_index, stop_index))
            for i in pbar:
                pbar.set_description(str(txn.stat()["entries"]))
                f = image_names[i]

                # TODO: skip 0.001 step
                x, y = imname_to_target(f)
                if int(x*100)%2 != 0:
                    continue
                if int(y*100)%2 != 0:
                    continue

                path = os.path.join(image_dir, f)

                # Load image
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # Process image
                _, img = process_single_image(f, img, target_size=resolution)

                # Serialize
                # Convert to NumPy array and serialize with msgpack
                img_array = np.array(img, dtype=np.uint8)
                img_bytes = msgpack.packb(img_array.tolist(), use_bin_type=True)
                # img_bytes = pickle.dumps(np.array(img))

                if use_compression:
                    # Compress
                    img_bytes = lz4.frame.compress(img_bytes, compression_level=6)

                txn.put(f.encode(), img_bytes)

                # Record key
                keys_file.write(f + "\n")

                if i%10000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        finally:
            try:
                txn.commit()
            except Exception as e:
                print(e)
            env.close()
            keys_file.close()

    # # Save image names
    # with open(os.path.join(lmdb_path, "keys.txt"), "+w") as keys_file:
    #     for l in image_names:
    #         keys_file.write(l + "\n")
    #     keys_file.close()


def read_image_from_lmdb(image_name: str, lmdb_path: str, decompress=True):
    # Open lmdb
    env = lmdb.open(lmdb_path, readonly=True)
    txn = env.begin()
    img_bytes = txn.get(image_name.encode())
    env.close()
    if img_bytes is None:
        raise KeyError(f"Image {image_name} not found in LMDB!")

    if decompress:
        img_bytes = lz4.frame.decompress(img_bytes)

    img = np.array(msgpack.unpackb(img_bytes, raw=False), dtype=np.uint8)

    return img


if __name__ == "__main__":
    datasource_dir = "H:/latest_real_data/real_data/real"
    output_path = "H:/real_512_0_002step.lmdb"
    # datasource_dir = "/mnt/h/latest_real_data/real_data/real"
    # output_path = "/mnt/h/real_512_0_002step.lmdb"
    # datasource_dir = "C:/Users/EVV13/Documents/multireflection/data/125x125_laser_x4_y6"
    # output_path = "H:/125x125_laser_x4_y6.lmdb"
    # create_lmdb_from_images(
    #     datasource_dir, output_path, stop_index=None, size=15 * 1024 * 1024 * 1024, use_compression=False
    # )
    # img = read_image_from_lmdb("x-0.01_y-0.49.jpg", output_path)
    # print(img.shape)

    # TEST all keys

    # keys = [s.replace("\n", "") for s in open(os.path.join(output_path, "keys.txt"), "r").readlines()]
    # for k in keys:
    #     read

    env = lmdb.open(output_path, readonly=True)
    with env.begin() as txn:
        length = txn.stat()['entries']
        print(length)
