from ptf.ClassesDict import ClassesDict
from pyolo.AnnotationXmlToYoloTxtConverter import AnnotationXmlToYoloTxtConverter


def main():
    work_dir = "d:/demo/training-yolo"

    classes_dict = ClassesDict()
    converter = AnnotationXmlToYoloTxtConverter(classes_dict)
    converter.train_image_dir = f"{work_dir}/images"
    converter.train_annotation_dir = f"{work_dir}/images"
    converter.yolo_txt_dir = f"{work_dir}/images"
    #converter.restore_train_data()
    converter.convert()
    converter.split_train_valid_data()
    converter.generate_yaml_file()


if __name__ == '__main__':
    main()
