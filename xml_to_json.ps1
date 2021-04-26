#Push-Location

$images_dir = "D:\VDisk\Network\OneDrive\Tensorflow\my-tf3\images"
#$images_dir = "D:\Demo\Yet-Another-EfficientDet-Pytorch-master\datasets\PostIt\train2021"
$dataset_dir = "D:\Demo\Yet-Another-EfficientDet-Pytorch-master\datasets\PostIt"

python ./scripts/xml_to_json.py `
    --images $images_dir `
    --dataset_dir $dataset_dir