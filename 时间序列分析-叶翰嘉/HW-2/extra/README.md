# 复现报告中的结果
1. 切换到 extra 目录
    ```shell
    cd extra
    ```
2. 运行得到训练损失和测试精度npy文件，包含使用EMA方法平滑前后的指标。
    ```shell
    python train.py
    ```

2. 运行读取 2 步骤中得到的npy文件，绘制使用EMA方法平滑前后的对比图。
    ```shell
    run codeblocks of plot.ipynb
    ```
