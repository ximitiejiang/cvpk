# cvpk

cvpk是一个计算机视觉包，主要用于支持计算机视觉/物体检测的应用
包含内容：
    + dataset数据集类: coco, voc
    + runner训练模块: runner, hooks
    + utils支持模块:


### Usage instruction
+ 安装:
    + git clone https://github.com/ximitiejiang/cvpk.git
    + run `sh compile.sh` to compile nms cuda extensions.
    + run `python3 setup.py install`  to install cvpk from source, or run `sh auto.sh` is also ok which will regenerate egg file. 
+ 卸载
    + 如果需要卸载干净，需要在安装时产生记录文件，然后基于记录文件进行删除
    + python3 setup.py install --record record.txt  # 获得安装记录文件
    + sudo rm $(cat record.txt)
    
+ 自定义修改：
    + 增加修改内容
    + 运行编译文件 `sh compile.sh`.
    + 运行自动生成和安装文件 `sh auto.sh`.


    