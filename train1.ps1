$work_dir = "d:/demo/trainging_yolo"
Push-Location

cd D:/demo/darknet

python train.py --device 0 --batch-size 16 `
    --data data/face.yaml `
    --cfg models/yolov4-csp_416.cfg `
    --weights '' `
    --name yolov4-csp

