#python detect.py --weights yolov5m.pt --img 640  --source D:\Demo\BCCD\test\images\BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg

$test_image = "D:\Demo\training-yolo\images\IMG_2169.jpg"

Push-Location

cd D:\demo\yolov5
python detect.py --weights `
    runs\train\exp16\weights\best.pt --img 640 `
    --conf 0.5 `
    --source $test_image

Pop-Location