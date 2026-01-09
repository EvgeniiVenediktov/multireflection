import lmdb
import os
from preprocess_images import process_single_image
from config import DATA_COLLECTION_CVT_TO_GRAYSCALE, DATA_COLLECTION_FINAL_RESOLUTION, TRAINING_IMAGE_RESOLUTION
import pickle
import cv2
import numpy as np
import msgpack
from tqdm import tqdm
import lz4.frame
from sklearn.model_selection import train_test_split
import torch
import io
from typing import Callable


def lmdb_bytes_to_torch_tensor(img_bytes: bytes) -> torch.Tensor:
    if img_bytes is None:
        raise ValueError("img_bytes is None")

    buf = io.BytesIO(img_bytes)
    buf.seek(0)
    obj = torch.load(buf, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and 'tensor' in obj and isinstance(obj['tensor'], torch.Tensor):
        t = obj['tensor']
    return t.contiguous()


def read_image_from_lmdb_tensor(image_name: str | list[str], lmdb_path: str, decompress: bool = True, dtype: torch.dtype | None = None, normalize: bool = True, device: str | torch.device = 'cpu') -> torch.Tensor | dict:
    """Simpler reader: fetch bytes from LMDB and convert with lmdb_bytes_to_torch_tensor.

    - If `image_name` is a string, returns a single `torch.Tensor` or None if missing.
    - If `image_name` is a list, returns a dict mapping key -> tensor or None.
    """
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        def _get_tensor(key: str):
            raw = txn.get(key.encode())
            if raw is None:
                return None
            return lmdb_bytes_to_torch_tensor(raw, decompress=decompress, dtype=dtype, normalize=normalize, device=device)

        if isinstance(image_name, str):
            res = _get_tensor(image_name)
            env.close()
            return res

        results = {k: _get_tensor(k) for k in image_name}
    env.close()
    return results

def imname_to_target(name:str) -> tuple[float, float]:
    """Parses image names of format x{x_value}_y{y_value}.jpg"""
    name = name.split('.jpg')[0]
    x, y = name.split("_")
    x = float(x[1:])
    if y[1] == '-':
        y = float(y[1:6])
    else:
        y = float(y[1:5])
    return x, y

def create_lmdb_from_images(
    image_dir, 
    lmdb_path, 
    start_index=0, 
    stop_index=None, 
    image_names: list[str] | None = None,
    size=None, 
    resolution=(512, 512), 
    use_compression=False,
    key_process=None, 
    key_filter=None, 
    keys_filename="keys.txt", 
    imread_mode = cv2.IMREAD_GRAYSCALE, 
    to_tensor: bool = True, 
    tensor_dtype: torch.dtype = torch.float32, 
    normalize: bool = True
):
    # Get filenames
    if image_names is None:
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
    keys_file = open(keys_filename, "+w")

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
                img = cv2.imread(path, imread_mode)

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
                img_bytes = None
                if to_tensor:
                    tensor = torch.from_numpy(img_array)
                    if tensor.ndim == 2:
                        tensor = tensor.unsqueeze(0)
                    else:
                        tensor = tensor.permute(2, 0, 1)
                    tensor = tensor.to(dtype=tensor_dtype)
                    if normalize:
                        tensor = tensor / 255.0

                    buf = io.BytesIO()
                    torch.save(tensor, buf)
                    img_bytes = buf.getvalue()

                else:
                    img_bytes = msgpack.packb(img_array.tolist(), use_bin_type=True)

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
                env.close()
                keys_file.close()
            except e:
                print(e)


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



def copy_selected_entries(
    src_lmdb_path: str,
    dst_lmdb_path: str,
    keys: list[str] | None = None,
    key_filter: Callable | None = None,
    map_size: int | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
    to_tensor: bool = False,
    tensor_dtype: torch.dtype = torch.float32,
    normalize: bool = False,
):
    """Copy selected entries from one LMDB to another.

    - If `keys` is provided, only those keys will be copied.
    - If `key_filter` is provided, it's used to filter keys (callable receiving key str).
    - If `to_tensor` is True, the data will be converted to a PyTorch tensor and stored
      using `torch.save` before optional compression.
    """
    src_env = lmdb.open(src_lmdb_path, readonly=True)
    src_txn = src_env.begin()

    # Determine keys to copy
    if keys is None:
        with src_env.begin() as tx:
            cursor = tx.cursor()
            all_keys = [k.decode() for k, _ in cursor]
    else:
        all_keys = list(keys)

    # Apply filter if present
    if key_filter is not None:
        all_keys = [k for k in all_keys if key_filter(k)]

    # Open destination env
    if map_size is None:
        map_size = 64 * 1024 * 1024 * 1024

    if stop_index is None:
        stop_index = len(all_keys)

    dst_env = lmdb.open(dst_lmdb_path, map_size=map_size)
    i = 0
    with dst_env.begin(write=True) as dst_txn:
        pbar = tqdm(range(start_index, stop_index))
        for i in pbar:
            k = all_keys[i]
            pbar.set_description(k)
            raw = src_txn.get(k.encode())
            if raw is None:
                continue

            data = raw
            # If converting to tensor, unpack msgpack then convert
            if to_tensor:
                # Decompress if needed
                try:
                    decompressed = lz4.frame.decompress(raw)
                except Exception:
                    decompressed = raw

                arr = np.array(msgpack.unpackb(decompressed, raw=False), dtype=np.uint8)
                tensor = torch.from_numpy(arr)
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                else:
                    tensor = tensor.permute(2, 0, 1)
                tensor = tensor.to(dtype=tensor_dtype)
                if normalize:
                    tensor = tensor / 255.0

                buf = io.BytesIO()
                torch.save(tensor, buf)
                data = buf.getvalue()


            dst_txn.put(k.encode(), data)

            if i%10000 == 0:
                dst_txn.commit()
                print(f"Copied {i} entries")
                dst_txn = dst_env.begin(write=True)

    src_env.close()
    dst_env.close()



def filter_002step(s:str) -> bool:
    """
    Leaves only even steps 
    False for filter out
    """
    x, y = imname_to_target(s)
    x = round(x*100)
    y = round(y*100)
    if not(x%2 == 0):
        return False
    if not(y%2 == 0):
        return False
    return True

def filter_004step(s:str) -> bool:
    """
    Leaves only 0.04 steps 
    False for filter out
    """

    x, y = imname_to_target(s)
    x = round(x*100)
    y = round(y*100)
    if not(x%4 == 0):
        return False
    if not(y%4 == 0):
        return False
    return True

def filter_016step(s:str) -> bool:
    """
    Leaves only 0.04 steps 
    False for filter out
    """

    x, y = imname_to_target(s)
    x = round(x*100)
    y = round(y*100)
    if not(x%16 == 0):
        return False
    if not(y%16 == 0):
        return False
    return True




def light_postfix_keyproc(s:str) -> str:
    return s[:-4] + "-light.jpg"

def main_light_postfix_keyproc(s:str) -> str:
    return s[:-4] + "-mainlight.jpg"

def newdirty_postfix_keyproc(s:str) -> str:
    return s[:-4] + "-newdirty.jpg"

if __name__ == "__main__":
    # datasource_dir = "/mnt/h/dark512"
    # output_path = "/mnt/h/real_512_0_001step.lmdb"
    

    # # datasource_dir = "/mnt/h/color_dark"
    # # output_path = "/mnt/e/color.lmdb"

    # imread_mode = cv2.IMREAD_COLOR_RGB
    # if DATA_COLLECTION_CVT_TO_GRAYSCALE:
    #     imread_mode = cv2.IMREAD_GRAYSCALE

    datasource_dir = "/mnt/h/dark512"
    output_path = "/mnt/h/black_512_0_001step_tensor.lmdb"

    # os.path.join(lmdb_path, keys_filename)
    
    image_names = []
    for s in open("./missing_keys.txt", "r").readlines():
            name = s.replace("\n", "")
            if len(name) == 0:
                continue
            image_names.append(name)

    # create_lmdb_from_images(
    #     datasource_dir, 
    #     output_path, 
    #     image_names=image_names,
    #     start_index=0,
    #     stop_index=None, 
    #     size=180 * 1024 * 1024 * 1024, 
    #     use_compression=False,
    #     resolution=DATA_COLLECTION_FINAL_RESOLUTION,
    #     # key_process=light_postfix_keyproc,
    # )
    
    
    # # TEST all keys
    
    env = lmdb.open(output_path, readonly=True)
    with env.begin() as txn:
        length = txn.stat()['entries']
        print(length)
    # #
    # exit()
    #
    # Dark:       228000
    # Dark+Light: 456000 
    # Dark+Light+MainLight: +57200 = 513200

    
    # Prepare keys
    # keys_fnames = ["keys_black.txt", "keys_light.txt", "keys_main_light.txt", "keys_newdirty.txt"]
    # keys_fnames = ["color_dark.txt", "color_mainlight_004.txt", "color_light.txt"]
    # keys_fnames = ["dark512.txt"]
    # # keys_fnames = ["color_mainlight_004.txt"]
    # keys = []
    # for fname in keys_fnames:
    #     for s in open(os.path.join(output_path, fname), "r").readlines():
    #         key = s.replace("\n", "")
    #         if filter_004step(key):
    #             keys.append(key)
    # write_split_keys(keys, output_path, train_fname="004_dark_train.txt", val_fname="004_dark_val.txt")

    src_path = "/mnt/e/real_512_0_001step.lmdb"
    dst_path = "/mnt/h/black_512_0_001step_tensor.lmdb"

    
    def get_all_fnames(image_dir:str) -> list[str]:
        return [
            str(f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
        ]

    def get_all_keys(lmdb_path:str) -> list[str]:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            keys = [bytes(k).decode() for k in cursor.iternext(keys=True, values=False)]
        env.close()
        return keys
    
    def missing_keys(src_keys:list[str], dst_keys:list[str]) -> list[str]:
        missing = []
        dst_key_set = set(dst_keys)
        for k in src_keys:
            if k not in dst_key_set:
                missing.append(k)
        return missing
    
    keys = get_all_fnames('/mnt/h/dark512')
    # keys_fname = "keys_black.txt"
    # for s in open(os.path.join(src_path, keys_fname), "r").readlines():
    #     key = s.replace("\n", "")
    #     keys.append(key)

    dst_keys = get_all_keys(dst_path)
    print("dst keys:", len(dst_keys))
    missing = missing_keys(keys, dst_keys)
    print("missing keys:", len(missing))
    with open("missing_keys.txt", "+w") as f:
        for k in missing:
            f.write(k + "\n")

    
    exit()

    # copy_selected_entries(
    #     src_path,
    #     dst_path,
    #     keys=keys,
    #     map_size= 264 * 1024 * 1024 * 1024,
    #     start_index=0,
    #     stop_index=10001,
    #     to_tensor=True,
    #     tensor_dtype=torch.float32,
    #     normalize=True,
    # )

    # env = lmdb.open(dst_path, readonly=True)
    # with env.begin() as txn:
    #     length = txn.stat()['entries']
    #     print(length)

