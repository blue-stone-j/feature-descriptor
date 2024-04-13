# feature-descriptor
Commonly used image feature and descriptors.

### compile and run
```bash
mkdir build
cd build
cmake ..
make
# compilation completed
./sift # run sift
./harris # run harris
```

### FAST
Features from Accelerated Segment Test

### harris
Harris Corner Detector. This code is based on [source](https://github.com/enazoe/toy/tree/master/harris). Useful reference [1](https://senitco.github.io/2017/06/18/image-feature-harris/).

### ORB-fast
Oriented FAST and Rotated BRIEF
Binary Robust Independent Elementary Features

### SIFT
Scale Invariant Feature Transform. This code is based on [source](https://developer.aliyun.com/article/1260641).
Useful reference [1](https://www.cnblogs.com/Alliswell-WP/p/SIFT.html) and [2](https://lsxiang.github.io/Journey2SLAM/computer_vision/SIFT/).

### SURF
Speeded Up Robust Features.