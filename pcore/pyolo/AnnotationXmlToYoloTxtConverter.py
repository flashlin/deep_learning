import glob
import os
import shutil

from ptf.AnnotationXml import AnnotationXml
from ptf.ClassesDict import ClassesDict
from py.Info import info
from py.Random import shuffle_split
from py.io import confirm_dirs


class AnnotationXmlToYoloTxtConverter:
    work_dir = "d:/demo/training-yolo"
    train_image_dir = "d:/demo/training-yolo/images"
    train_annotation_dir = "d:/demo/training-yolo/images"
    yolo_txt_dir = "d:/demo/training-yolo/images"

    def __init__(self, classes_dict: ClassesDict):
        self.classes_dict = classes_dict

    def convert(self):
        data_counter = 0
        filters = f"{self.train_annotation_dir}/*.xml"
        print(f"filters = {filters}")
        for xml_file_path in glob.glob(filters):
            img_name = os.path.basename(xml_file_path).split('.')[0]
            img_name_path = f"{self.train_image_dir}/{img_name}.jpg"
            if not os.path.isfile(img_name_path):
                continue
            print(".", end="")
            annotation_xml_file = AnnotationXml(xml_file_path)

            img_info = []
            for item in annotation_xml_file:
                filename, img_width, img_height, label_name, bbox = item
                xmin, ymin, xmax, ymax = bbox

                x = (xmin + (xmax - xmin) / 2) * 1.0 / img_width
                y = (ymin + (ymax - ymin) / 2) * 1.0 / img_height
                w = (xmax - xmin) * 1.0 / img_width
                h = (ymax - ymin) * 1.0 / img_height

                classes_id = self.classes_dict.add_classes_name(label_name) - 1
                # print(f"id={classes_id}")

                line = ' '.join([str(classes_id), str(x), str(y), str(w), str(h)])
                img_info.append(line)

                # copy image to yolo path and rename
                img_name = os.path.basename(filename)
                # print(f"[{data_counter}] {img_name} {line}")
                # img_path = os.path.join(self.train_image_dir, img_name)
                # if not os.path.exists(target_file_path):
                #    shutil.copyfile(img_path, target_file_path)

            yolo_txt_file_path = f"{self.yolo_txt_dir}/{self.generate_yolo_file_name(data_counter, img_name)}"
            #with open(yolo_path + str(data_counter) + '.txt', 'w') as f:
            with open(yolo_txt_file_path, 'w') as f:
                f.write('\n'.join(img_info))
            data_counter += 1

        print("")
        print(f'the {data_counter} file is processed')

    def generate_yolo_file_name(self, data_counter, image_name):
        name = image_name.split('.')[0]
        # img_format = image_name.split('.')[1]  # jpg or png
        # return str(self.data_counter) + '.' + img_format
        return f"{name}.txt"

    def split_train_valid_data(self):
        self.restore_train_data()
        yolo_txt_files = glob.glob(f"{self.yolo_txt_dir}/*.txt")
        info(f"yolo_txt_files = {len(yolo_txt_files)}")
        train_yolo_txt_files, valid_yolo_txt_files = shuffle_split(yolo_txt_files)
        info(f"train_size={len(train_yolo_txt_files)} valid_size={len(valid_yolo_txt_files)}")
        self.clean_yolo_txt_files()
        self.move_image_files(train_yolo_txt_files, "train")
        self.move_yolo_txt_files(train_yolo_txt_files, "train")
        self.move_image_files(valid_yolo_txt_files, "valid")
        self.move_yolo_txt_files(valid_yolo_txt_files, "valid")

    def move_image_files(self, yolo_txt_files, train_or_valid):
        target_dir = f"{self.work_dir}/{train_or_valid}/images"
        confirm_dirs(target_dir)
        for txt_file in yolo_txt_files:
            txt_name = os.path.basename(txt_file).split('.')[0]
            source_file = f"{self.work_dir}/images/{txt_name}.jpg"
            target_file = f"{target_dir}/{txt_name}.jpg"
            if os.path.isfile(target_file):
                continue
            info(f"{source_file} -> {target_file}")
            shutil.move(source_file, target_file)

    def move_yolo_txt_files(self, yolo_txt_files, train_or_valid):
        target_dir = f"{self.work_dir}/{train_or_valid}/labels"
        confirm_dirs(target_dir)
        for txt_file in yolo_txt_files:
            txt_name = os.path.basename(txt_file)
            target_file = f"{target_dir}/{txt_name}"
            if os.path.isfile(target_file):
                continue
            info(f"{txt_file} -> {target_file}")
            shutil.move(txt_file, target_file)

    def restore_train_valid_data(self, train_or_valid):
        for source_file in glob.glob(f"{self.work_dir}/{train_or_valid}/images/*.jpg"):
            source_name = os.path.basename(source_file)
            target_file = f"{self.train_image_dir}/{source_name}"
            shutil.move(source_file, target_file)

    def restore_train_data(self):
        self.restore_train_valid_data("train")
        self.restore_train_valid_data("valid")


    def clean_yolo_txt_files(self):
        for file in glob.glob(f"{self.work_dir}/train/labels/*.*"):
            os.remove(file)
        for file in glob.glob(f"{self.work_dir}/train/*.*"):
            os.remove(file)
        for file in glob.glob(f"{self.work_dir}/valid/labels/*.*"):
            os.remove(file)
        for file in glob.glob(f"{self.work_dir}/valid/*.*"):
            os.remove(file)

    def generate_yaml_file(self):
        yolo_yaml_file_path = f"{self.work_dir}/yolov5.yaml"
        with open(yolo_yaml_file_path, 'w') as f:
            f.write(f"train: {self.work_dir}/train/images\n")
            f.write(f"val: {self.work_dir}/valid/images\n")
            f.write(f"nc: {len(self.classes_dict.names)}\n")
            f.write(f"names: [\n")
            for idx in range(len(self.classes_dict.names)):
                name = self.classes_dict.names[idx]
                f.write(f"'{name}'")
                if idx < len(self.classes_dict.names)-1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")
