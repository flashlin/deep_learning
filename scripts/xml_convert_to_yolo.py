import shutil

from ptf.ClassesDict import ClassesDict
from py.Info import info

from py.io import confirm_dirs
from pyolo.Utils import convert_xml_to_yolo_format, write_face_names_file, write_face_data_file, setup_yolo_cfg_file, \
    clean_yolo_path, update_anchors_txt_file


class ConvertAnnotationXmlToYoloHelper:
    classes_dict = ClassesDict()
    work_dir = "d:/demo/training-yolo"
    train_img = f"{work_dir}/your/images"
    train_annotation = f"{work_dir}/your/annotations"

    #train_img = "D:/VDisk/Network/OneDrive/Tensorflow/RealTimeObjectDetection/workspace/images/raw"
    #train_annotation = train_img

    def convert(self):
        yolo_path = confirm_dirs(f"{self.work_dir}/yolo_data/")
        cfg_dir = confirm_dirs(f"{self.work_dir}/cfg")
        write_train_txt = f"{cfg_dir}/train.txt"
        write_val_txt = f"{cfg_dir}/val.txt"
        clean_yolo_path(yolo_path)

        convert_xml_to_yolo_format(self.train_img, self.train_annotation, yolo_path)

        face_names_file_path = f"{cfg_dir}/face.names"
        write_face_names_file(face_names_file_path, self.classes_dict)


        face_data_file_path = f"{cfg_dir}/face.data"
        classes_num = len(self.classes_dict.names)

        write_face_data_file(cfg_dir, self.classes_dict)
        cfg_file_path = f"{cfg_dir}/yolov4-tiny-obj.cfg"
        shutil.copyfile(f"{self.work_dir}/../darknet/cfg/yolov4-tiny-custom.cfg", cfg_file_path)

        setup_yolo_cfg_file(cfg_file_path, classes_num)


#compute_anchors(work_dir)
import argparse

def run_update_anchors(args, **config):
    info(f"update anchars")
    update_anchors_txt_file(f"{config.work_dir}/../darknet/anchors.txt", config.cfg_file_path)

def run_convert_annotation_xml_to_yolo_data(args, **config):
    info(f"convert annotation xml to yolo data")
    p = ConvertAnnotationXmlToYoloHelper()
    p.convert()


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument('--toggle', '--no-toggle', dest='toggle', action=NegateAction, nargs=0)
    ap.add_argument('--convert', dest='toggle', nargs=0)
    ap.add_argument('--update_anchors', dest='toggle', nargs=0)
    args = ap.parse_args()

    work_dir = "d:/demo/training_yolo",
    cfg_dir = f"{work_dir}/cfg",
    cfg_file_path = f"{cfg_dir}/yolov4-tiny-obj.cfg",
    config = {
        work_dir,
        cfg_dir,
        cfg_file_path
    }

    if args.update_anchors:
        run_update_anchors(args, config)
        return

    if args.convert:
        run_convert_annotation_xml_to_yolo_data(args, config)
        return

    info(f"--convert")
    info(f"--update_anchors")


if __name__ == '__main__':
    main()



