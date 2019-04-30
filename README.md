# cvpk

cvpk is a computer vision package, mainly used to support deep learning object detection.

### Usage instruction
+ if you want to install cvpk, you need firstly download it from github:
    + git clone https://github.com/ximitiejiang/cvpk.git
    + run `sh compile.sh` to compile nms cuda extensions.
    + run `python3 setup.py install`  to install cvpk from source, or run `sh auto.sh` is also ok which will regenerate egg file. 
    
+ if you modified cvpk and want to use it again, you need to regenerate egg file and re-install it directly
    + run `sh compile.sh` if you modified the nms cuda extensions, otherwise jump to next step.
    + run `sh auto.sh` to regenerate egg file and install from source.

    