$width = 416
$height = 416
$classes = 3

$filters = ($classes + 5) * 3

sed -i "8s/512/$width/" models/yolov4-csp.cfg
sed -i "9s/512/$height/" models/yolov4-csp.cfg
sed -i "1022s/255/$filters/" models/yolov4-csp.cfg
sed -i "1029s/80/$classes/" models/yolov4-csp.cfg
sed -i "1131s/255/$filters/" models/yolov4-csp.cfg
sed -i "1138s/80/$classes/" models/yolov4-csp.cfg
sed -i "1240s/255/$filters/" models/yolov4-csp.cfg
sed -i "1247s/80/$classes/" models/yolov4-csp.cfg
# 查看修改後的參數
$ sed -n -e 8p -e 9p -e 1022p -e 1029p -e 1131p -e 1138p -e 1240p -e 1247p models/yolov4-csp.cfg