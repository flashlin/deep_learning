Push-Location
cd d:/demo
cd yolov5

# python train.py --img 640 --batch 16 --epochs 5 --data bccd-data.yaml --weights yolov5s.pt
# python train.py --img 640 --batch 16 --resume --data yolov5.yaml --weights yolov5s.pt
python train.py --img 640 --batch 3 --epochs 20 --data yolov5.yaml --weights yolov5x.pt #--weights runs\train\exp17\weights\best.pt



Write-Host "d:/demo/yolov5"
Write-Host "tensorboard --logdir=runs"
Pop-Location