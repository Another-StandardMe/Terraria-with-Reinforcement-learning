import os

yolo_txt_dir = "E:/terraria_project/GAME_ENV_DATASET/labels/train"

def remove_label_3_from_txt(file_path):
    """ 从 YOLO txt 文件中删除 label 为 3 的行 """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # 过滤掉 label 为 3 的行
    new_lines = [line for line in lines if not line.startswith("3 ")]

    # 重新写入文件
    with open(file_path, "w") as f:
        f.writelines(new_lines)

# 遍历目录，处理所有 txt 文件
for filename in os.listdir(yolo_txt_dir):
    if filename.endswith(".txt"):  # 只处理 txt 文件
        file_path = os.path.join(yolo_txt_dir, filename)
        remove_label_3_from_txt(file_path)
        print(f"处理完成: {file_path}")

print("所有文件处理完毕！")
