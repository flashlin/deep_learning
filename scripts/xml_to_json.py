import argparse
import glob
import json
import os.path
import shutil

from PIL import Image

from ptf.AnnotationXml import AnnotationXml
from ptf.ClassesDict import ClassesDict
from py.Info import info
from py.Random import shuffle_split


def parse_args():
    parser = argparse.ArgumentParser(description='Annotation Xml File convert tool')
    parser.add_argument('--images', metavar='DIR', help='path to images')
    parser.add_argument('--dataset_dir', metavar='DIR', help='path to dataset')
    args = parser.parse_args()
    return args


def get_image_size(image_path):
    image = Image.open(image_path)
    return image.size


def move_image_files_to_dataset_folder(images, target_dir):
    for image in images:
        img_dir = os.path.dirname(image)
        img_filename = os.path.basename(image)
        img_name = img_filename.split('.')[0]
        anno_xml_file_path = f"{img_dir}/{img_name}.xml"
        if not os.path.isfile(anno_xml_file_path):
            continue
        train_image_file_path = f"{target_dir}/{img_filename}"
        train_anno_file_path = f"{target_dir}/{img_name}.xml"
        shutil.move(image, train_image_file_path)
        shutil.move(anno_xml_file_path, train_anno_file_path)


def restore_images_folder(train_dir, images_dir):
    for file in glob.glob(f"{train_dir}/*.*"):
        filename = os.path.basename(file)
        target_file = f"{images_dir}/{filename}"
        shutil.move(file, target_file)


def main():
    args = parse_args()
    images_dir = args.images
    dataset_dir = args.dataset_dir

    classes_dict = ClassesDict()
    train_images_dir = f"{dataset_dir}/train2021"
    val_images_dir = f"{dataset_dir}/val2021"

    info(f"images_dir = {images_dir}")
    images = glob.glob(f"{images_dir}/*.jpg")
    train_images, val_images = shuffle_split(images)

    info(f"train_images = {len(train_images)}")
    info(f"val_images = {len(val_images)}")

    restore_images_folder(train_images_dir, images_dir)
    restore_images_folder(val_images_dir, images_dir)
    move_image_files_to_dataset_folder(train_images, train_images_dir)
    move_image_files_to_dataset_folder(val_images, val_images_dir)

    generate_annotations_files(classes_dict, dataset_dir, train_images_dir, "train2021")
    generate_annotations_files(classes_dict, dataset_dir, val_images_dir, "val2021")


def generate_annotations_files(classes_dict, dataset_dir, train_images_dir, train_name):
    json_data = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    anno_id = 0
    output_name = train_name
    for annotation_xml_file in glob.glob(f"{train_images_dir}/*.xml"):
        image_name = os.path.basename(annotation_xml_file).split('.')[0]
        image_filename = f"{image_name}.jpg"
        image_path = f"{train_images_dir}/{image_filename}"
        image_width, image_height = get_image_size(image_path)
        image_id = len(json_data["images"]) + 1
        image_json = {
            "file_name": f"{image_filename}",
            "height": image_height,
            "width": image_width,
            "id": image_id
        }
        json_data["images"].append(image_json)
        annotation_xml = AnnotationXml(annotation_xml_file)
        anno_id = anno_id + 1
        for ann in annotation_xml:
            _, _, _, label_name, bbox = ann
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            anno_json = {
                "area": 26091,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, w, h],
                "category_id": classes_dict.add_classes_name(label_name),
                "id": anno_id,
                "ignore": 0,
                "segmentation": []
            }
            json_data["annotations"].append(anno_json)
    for label_name in classes_dict.names:
        category_json = {
            "supercategory": "none",
            "id": classes_dict.dict[label_name],
            "name": label_name
        }
        json_data["categories"].append(category_json)
    with open(f"{dataset_dir}/annotations/instances_{output_name}.json", 'w') as outfile:
        json.dump(json_data, outfile)


if __name__ == '__main__':
    main()
