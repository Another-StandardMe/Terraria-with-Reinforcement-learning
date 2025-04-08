# This code converts XML annotation files into YOLO-format TXT labels.
# It reads object bounding boxes from XML files, normalizes them, and saves the results in YOLO format.
# The converted labels can be used for training YOLO object detection models.

import os
import xml.etree.ElementTree as ET

class_mapping = {
    "Eye_of_Cthulhu": 0,
    "King_Slime": 1,
    "Player": 2,
    "Cursor": 3
}

def convert_xml_to_yolo(xml_file, output_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    txt_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")

    with open(txt_filename, "w") as txt_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            bbox_width = (xmax - xmin) / image_width
            bbox_height = (ymax - ymin) / image_height

            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    print(f"Converted: {txt_filename}")

def batch_convert_xml_to_yolo(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    xml_files = [f for f in os.listdir(input_folder) if f.endswith(".xml")]

    for xml_file in xml_files:
        xml_path = os.path.join(input_folder, xml_file)
        convert_xml_to_yolo(xml_path, output_folder)

input_folder = "E:/terraria_project/daytime/labels/train"
output_folder = "E:/terraria_project/daytime/labels/train"

batch_convert_xml_to_yolo(input_folder, output_folder)
print("All XML files have been converted!")
