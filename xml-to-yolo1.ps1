Push-Location
$work_dir = "d:/demo/training"

python ./scripts/xml_to_yolo.py

#sed -n -e 1028p -e 1137p -e 1246p models/yolov4-csp_416.cfg

Pop-Location