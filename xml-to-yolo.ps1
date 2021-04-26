Push-Location
$work_dir = "d:/demo/training-yolo"

python ./scripts/xml_convert_to_yolo.py --convert

#sed -n -e 1028p -e 1137p -e 1246p models/yolov4-csp_416.cfg

cd "$($work_dir)/../darknet"
D:\Demo\darknet\build\darknet\x64\darknet_no_gpu.exe detector `
    calc_anchors $work_dir/cfg/face.data `
    -num_of_clusters 6 `
    -width 416 -height 416
    #-showpause


#下載 yolov4-tiny Darknet 官方事先訓練好的 weight (yolov4-tiny.conv.29) 放入 Face_detection/cfg 中，就可以開始做訓練啦!

Pop-Location