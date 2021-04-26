$env:PATH = "D:\Demo\darknet\build\darknet\x64"
$work_dir = "d:/demo/training-yolo"

$darknet = "D:\Demo\darknet\build\darknet\x64\darknet_no_gpu.exe"
# $darknet detector train `
#     $work_dir/cfg/face.data `
#     $work_dir/cfg/yolov4-tiny-obj.cfg `
#     $work_dir/cfg/yolov4-tiny.conv.29 `
#     -dont_show


#訓練好的 weights 會放在 cfg/weights 裡
#然後打開 yolov4-tiny-obj.cfg
#將net 裡Testing 的 batch, subdivisions 註解刪掉
#並註解 Training 的 batch, subdivisions (如下圖)


darknet_no_gpu.exe detector train `
    $work_dir/cfg/face.data `
    $work_dir/cfg/yolov4-tiny-obj.cfg `
    $work_dir/cfg/yolov4-tiny.conv.29 `
    -dont_show