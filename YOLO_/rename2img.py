# This code renames all JPG files in a specified folder with a sequential naming format.
# It ensures consistent image naming for dataset organization and model training.
# The renamed files follow the format: <prefix><index>.jpg (e.g., terrariaENV_tr001.jpg).

import os

folder_path = "E:/terraria_project/GAME_ENV_DATASET/images/train"
new_name_format = "terrariaENV_tr"

jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])

for i, filename in enumerate(jpg_files, start=1):
    old_path = os.path.join(folder_path, filename)
    new_filename = f"{new_name_format}{i:03d}.jpg"
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)
    print(f" {filename} → {new_filename}")

print("Renaming completed!")
