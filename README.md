# 文件结构

1. data：数据集

2. demo：opencv 库实现摄像头实时推断

3. experiments：每次修参（不同名字的文件夹）保留的模型参数和结果，可用于可视化和复现

4. images：可视化和过程产生的图片

5. models：存的自定义cnn和class

6. train.py：训练主文件

   ```python
   python train.py -h 
   ```

   获得与设置相关参数信息

   ```
   python train.py
   ```

   产生一个名为try的实验数据（均为默认参数）

7. util.py：存放一些函数定义
8. infer.py：利用单张测试样例进行推断