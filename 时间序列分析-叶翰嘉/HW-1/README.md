# 复现报告中的结果
1. 切换到 code 目录
    ```shell
    cd code
    ```
2. 运行得到 result.xlsx，包含不同方法在不同变换下的性能指标；还会得到不同模型预测序列与真实序列的曲线图。
    ```shell
    python main.py
    ```
3.  运行得到 result2.xlsx，包含不同周期下$Naive_s$模型的性能指标；还会得到不同性能随周期变化的曲线图。
    ```shell
    python best_period.py
    ```
