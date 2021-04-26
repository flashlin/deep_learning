Push-Location
cd ..
git clone https://github.com/AlexeyAB/darknet

#sed -i "s/GPU=0/GPU=1/g" darknet/Makefile
#sed -i "s/CUDNN=0/CUDNN=1/g" darknet/Makefile
#sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/g" darknet/Makefile
sed -i "s/OPENCV=0/OPENCV=1/g" darknet/Makefile
cd darknet
make