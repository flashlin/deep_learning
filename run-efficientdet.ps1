Push-Location

cd D:\Demo\Yet-Another-EfficientDet-Pytorch-master
python train.py -c 2 -p PostIt --batch_size 8 `
    --lr 1e-3 `
    --num_epochs 10 `
    --load_weights weights/efficientdet-d7.pth `
    --head_only True `
    --saved_path checkpoints

Pop-Location