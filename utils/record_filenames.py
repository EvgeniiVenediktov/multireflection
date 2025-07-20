import os

output_file_path = "/mnt/h/color.lmdb/color_light.txt"
img_dir = "/mnt/h/color_light"

image_names = [
        str(f) for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))
    ]

with open(output_file_path, "+w") as f:
    for n in image_names:
        f.write(n[:-4] + "-light.jpg"+"\n")

