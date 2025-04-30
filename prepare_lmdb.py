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
from sklearn.model_selection import train_test_split

def imname_to_target(name:str) -> tuple[float]:
    """Parses image names of format x{x_value}_y{y_value}.jpg"""
    name = name.split('.jpg')[0]
    x, y = name.split("_")
    x = float(x[1:])
    y = float(y[1:5])
    return x, y

def create_lmdb_from_images(
    image_dir, lmdb_path, start_index=0, stop_index=None, size=None, resolution=(512, 512), use_compression=True,
    key_process=None, key_filter=None
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

                if key_filter is not None:
                    if not key_filter(f):
                        continue

                path = os.path.join(image_dir, f)

                # Load image
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if key_process is not None:
                    f = key_process(f)

                # Process image
                try:
                    _, img = process_single_image(f, img, target_size=resolution)
                except Exception as e:
                    print(e)
                    print(f)
                    continue

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


def write_split_keys(keys:list[str], path, filter:callable = None, val_share=0.2, train_fname="train_keys.txt", val_fname="val_keys.txt") -> None:
    # Filter keys
    if filter is not None:
        new_keys = []
        for k in keys:
            if not filter(k):
                continue
            new_keys.append(k)
        keys = new_keys

    # Split
    train, val = train_test_split(keys, test_size=val_share)
    print("train len:", len(train))
    print("val len:", len(val))
    # Write
    with open(os.path.join(path, train_fname), "+w") as keys_file:
        for l in train:
            keys_file.write(l + "\n")
        keys_file.close()
    with open(os.path.join(path, val_fname), "+w") as keys_file:
        for l in val:
            keys_file.write(l + "\n")
        keys_file.close()


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



def filter_002step(s:str) -> bool:
    """False for filter out"""
    x, y = imname_to_target(s)
    x = round(x*100)
    y = round(y*100)
    if not(x%2 == 0):
        return False
    if not(y%2 == 0):
        return False
    return True


def light_postfix_keyproc(s:str) -> str:
    return s[:-4] + "-light.jpg"

if __name__ == "__main__":
    datasource_dir = "/mnt/h/latest_real_data/light"
    output_path = "/mnt/h/real_512_0_001step.lmdb"

    # create_lmdb_from_images(
    #     datasource_dir, 
    #     output_path, 
    #     stop_index=None, 
    #     size=120 * 1024 * 1024 * 1024, 
    #     use_compression=False, 
    #     key_process=light_postfix_keyproc
    # )
    
    
    # TEST all keys

    # env = lmdb.open(output_path, readonly=True)
    # with env.begin() as txn:
    #     length = txn.stat()['entries']
    #     print(length)

    # Dark:       228000
    # Dark+Light: 456000 

    # Prepare keys
    # keys = [s.replace("\n", "") for s in open(os.path.join(output_path, "keys.txt"), "r").readlines()]
    keys_fnames = ["keys_black.txt", "keys_light.txt"]
    keys = []
    for fname in keys_fnames:
        for s in open(os.path.join(output_path, fname), "r").readlines():
            keys.append(s.replace("\n", ""))
    write_split_keys(keys, output_path, train_fname="mixed_keys_train.txt", val_fname="mixed_keys_val.txt")

