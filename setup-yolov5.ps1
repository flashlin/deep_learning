Push-Location

$trainingConfig = ConvertFrom-Json $env:_trainingConfig

cd $trainingConfig.demoDir

git clone https://github.com/ultralytics/yolov5
#pip install -U -r yolov5/requirements.txt

cd yolov5
python train.py --img 640 --batch 16 --epochs 5 --data bccd-data.yaml --weights yolov5s.pt