import glob
import math
import os
import re
import shutil

from ptf.AnnotationXml import AnnotationXml
from ptf.ClassesDict import ClassesDict
from py.Info import info
from py.io import sed


def convert_xml_to_yolo_format(train_img, train_annotation, yolo_path):
    global classes_dict
    now_path = os.getcwd()
    data_counter = 0

    for data_file in glob.glob(f"{train_annotation}/*.xml"):
        # data_file_path = os.path.join(train_annotation, data_file)
        # print(".", end="")
        annotation_xml_file = AnnotationXml(data_file)

        img_info = []
        for item in annotation_xml_file:
            filename, img_width, img_height, label_name, bbox = item
            xmin, ymin, xmax, ymax = bbox

            x = (xmin + (xmax - xmin) / 2) * 1.0 / img_width
            y = (ymin + (ymax - ymin) / 2) * 1.0 / img_height
            w = (xmax - xmin) * 1.0 / img_width
            h = (ymax - ymin) * 1.0 / img_height

            classes_id = classes_dict.add_classes_name(label_name)
            # print(f"id={classes_id}")

            line = ' '.join([str(classes_id), str(x), str(y), str(w), str(h)])
            img_info.append(line)

            # copy image to yolo path and rename
            img_name = os.path.basename(filename)
            # print(f"[{data_counter}] {img_name} {line}")
            img_path = os.path.join(train_img, img_name)
            img_format = img_name.split('.')[1]  # jpg or png
            target_file_path = yolo_path + str(data_counter) + '.' + img_format
            if not os.path.exists(target_file_path):
                shutil.copyfile(img_path, target_file_path)

        with open(yolo_path + str(data_counter) + '.txt', 'w') as f:
            f.write('\n'.join(img_info))
        data_counter += 1

    print("")
    print('the file is processed')

    # create train and val txt
    path = os.path.join(now_path, yolo_path)
    datasets = []
    for idx in os.listdir(yolo_path):
        if not idx.endswith('.txt'):
            idx_path = path + idx
            datasets.append(idx_path)

    cfg_dir = f"{yolo_path}../cfg"
    write_train_txt = f"{cfg_dir}/train.txt"
    write_val_txt = f"{cfg_dir}/val.txt"
    len_datasets = math.floor(len(datasets) * 0.8)
    with open(write_train_txt, 'w') as f:
        f.write('\n'.join(datasets[0:len_datasets]))

    with open(write_val_txt, 'w') as f:
        f.write('\n'.join(datasets[len_datasets:]))


def write_face_names_file(face_names_file_path: str, classes_dict: ClassesDict):
    with open(face_names_file_path, 'w') as f:
        for name in classes_dict.names:
            f.write(f"{name}\n")
    return face_names_file_path


def write_face_data_file(cfg_dir: str, classes_dict: ClassesDict):
    classes_num = len(classes_dict.names)
    train_txt_path = f"{cfg_dir}/train.txt"
    val_txt_path = f"{cfg_dir}/val.txt"
    face_names_file_path = f"{cfg_dir}/face.names"
    with open(face_data_file_path, 'w') as f:
        f.write(f"classes={classes_num}\n")
        f.write(f"train={train_txt_path}\n")
        f.write(f"valid={val_txt_path}\n")
        f.write(f"names={face_names_file_path}\n")
        f.write(f"backup={cfg_dir}/weights\n")
    return classes_num


def setup_yolo_cfg_file(cfg_file_path, classes_num, features_num=5):
    # yolov4 偵測的濾鏡(filter) 大小為 (C+5)*B
    # B 是每個Feature Map可以偵測的bndBox數量，這裡設定為3
    # 5 是bndBox輸出的5個預測值: x,y,w,h 以及 Confidence
    # C 是類別數量
    # filters=(classes + 5)*3  # 因為是一個類別，所以filters更改為 18
    # classes=1  #人臉偵測只有一個類別
    filters = (classes_num + features_num) * 3
    sed(cfg_file_path, f"212s/255/{filters}/")
    sed(cfg_file_path, f"220s/80/{classes_num}/")
    sed(cfg_file_path, f"263s/255/{filters}/")
    sed(cfg_file_path, f"269s/80/{classes_num}/")


def clean_yolo_path(yolo_path):
    lsdir = os.listdir(yolo_path)
    for name in lsdir:
        if name.endswith('.txt') or name.endswith('.jpg') or name.endswith('.png'):
            os.remove(os.path.join(yolo_path, name))


def update_anchors_txt_file(darknet_anchors_txt_path, cfg_file_path):
    #anchors_txt_path = f"{work_dir}/../darknet/anchors.txt"
    with open(darknet_anchors_txt_path, 'r') as f:
        anchors_result = f.readline()
        info(f"{anchors_result}")

    regex = re.compile(r'anchors = (.+)')
    match = regex.search(anchors_result)
    if not match:
        raise Exception("fail")
    anchors_txt = match.group(1)
    sed(cfg_file_path, f"219s/anchors = .+/anchors = {anchors_txt}")
    sed(cfg_file_path, f"268s/anchors = .+/anchors = {anchors_txt}")