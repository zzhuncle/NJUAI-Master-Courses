# Contrastive Learning for Multivariate Time Series Forecasting

This repo is the Pytorch implementation of C2Linear: [Contrastive Learning for Multivariate Time Series Forecasting](../报告.pdf).

## C2Linear
### C2Linear family

| Models    | Code files                          |
| --------- | ----------------------------------- |
| C2Linear  | `code/C2Linear/models/C2Linear.py`  |
| C2NLinear | `code/C2Linear/models/C2NLinear.py` |
| C2DLinear | `code/C2Linear/models/C2DLinear.py` |

The implement of contrastive regularization term is in file `code/C2Linear/exp/exp_main.py`.

```python
    # @zhuangzh 对比学习正则化项
    y = embedding
    N = self.args.batch_size
    y = y.reshape(N, -1)
    D = y.shape[-1]
    y = y - y.mean(dim=0)

    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_y))


    cov_y = (y.T @ y) / (N - 1)
    cov_loss = off_diagonal(cov_y).pow_(2).sum() / D
    reg_loss = (0.0001 * std_loss + 0.0001 * cov_loss) * 96 / D
    loss += reg_loss
    # @zhuangzh 
```

### Reproduction
For example:

To train the **C2Linear** on **Exchange-Rate dataset**, you can use the script `scripts/EXP-LongForecasting/Linear/exchange_rate.sh`:
```
sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh
```
It will start to train C2DLinear by default, the results will be shown in `logs/LongForecasting`. You can specify the name of the model in the script. (C2Linear, C2DLinear, C2NLinear)

For **all nine real-world datasets**, you can run the following command.

```
sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh
sh scripts/EXP-LongForecasting/Linear/electricity.sh
sh scripts/EXP-LongForecasting/Linear/etth1.sh
sh scripts/EXP-LongForecasting/Linear/etth2.sh
sh scripts/EXP-LongForecasting/Linear/ettm1.sh
sh scripts/EXP-LongForecasting/Linear/ettm2.sh
sh scripts/EXP-LongForecasting/Linear/ili.sh
sh scripts/EXP-LongForecasting/Linear/traffic.sh
sh scripts/EXP-LongForecasting/Linear/weather.sh
```

