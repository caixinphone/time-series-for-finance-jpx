# JPX Tokyo Stock Exchange Prediction Baseline

这个项目用于完成一个真实的 Kaggle 金融时间序列竞赛闭环：下载 JPX 数据、解释任务、训练 baseline、离线验证，并生成 `submission.csv`。


## 本项目选择的真实竞赛

真实竞赛是 `JPX Tokyo Stock Exchange Prediction`。任务是每天为约 2000 只日本股票预测未来收益，并输出每只股票的排名：

- `Rank = 0` 表示当天最看好的股票。
- Kaggle 评分基于每日多空组合的 Sharpe Ratio：做多排名前 200，做空排名后 200，观察日收益价差的均值/波动。
- 这是历史 Code Competition；本地完成标准是跑通训练、验证和生成符合格式的提交文件。

## 快速运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/kaggle auth login
.venv/bin/kaggle competitions download jpx-tokyo-stock-exchange-prediction -p data/jpx
unzip -q -o data/jpx/jpx-tokyo-stock-exchange-prediction.zip -d data/jpx
.venv/bin/python jpx_baseline.py --data-root data/jpx --output outputs/submission.csv
```

快速试跑可以加采样参数：

```bash
.venv/bin/python jpx_baseline.py --max-train-rows 300000
```

## Baseline 方法

脚本只使用主价格表和股票静态信息：

- 价格特征：开高低收、成交量、调整后收盘价、日内涨跌、最高最低价差。
- 动量/反转特征：1、2、5、10、20 日收益率。
- 波动特征：过去 5、10、20 日收益率均值和标准差。
- 截面特征：行业代码、指数规模、流通市值、发行股数、Universe0。
- 模型：优先用 LightGBM 回归预测 `Target`，再按每个交易日预测值降序生成 `Rank`。
- 如果 macOS 缺少 `libomp.dylib`，脚本会自动回退到 sklearn `HistGradientBoostingRegressor`，保证完整流程能跑通。

## 当前最好提交

- Kaggle kernel：`caixin030703/jpx-local-baseline-submit`
- Kernel version：`24`
- Submission ref：`52720631`
- Public score：`0.366`
- Private score：`0.366`
- 主要改进：将公开第 2 名 LightGBM 方案与公开第 4 名规则模型做 rank ensemble。LightGBM 使用 20/40/60 日收益率、波动率和均线偏离特征；规则模型按调整后 1 日收益反向排序并惩罚股息日。

已验证过的主要线上版本：

- v9：滚动时间序列特征 + sklearn 逆向 ensemble，`0.130 / 0.130`。
- v13：多随机种子的浅层 LightGBM 差分 ensemble，`0.157 / 0.157`。
- v14：alpha-tail LightGBM，只训练每日收益两端股票，`0.165 / 0.165`。
- v16：单特征 `close_diff1` LightGBM，`0.214 / 0.214`。
- v18：公开第 4 名规则模型，按 1 日收益反向排序并惩罚股息日，`0.344 / 0.344`。
- v21：公开第 2 名 LightGBM 方案复刻，`0.356 / 0.356`。
- v24：LightGBM + 规则模型 rank ensemble，`0.366 / 0.366`。

输出文件：

- `outputs/submission.csv`：本地生成的提交文件。
- `outputs/metrics.json`：验证 Sharpe、RMSE、提交文件检查。
- `outputs/feature_importance.csv`：LightGBM 特征重要性。

## 注意

这不是投资建议。该 baseline 是为了复现竞赛流程和建立可迭代起点，后续提分可以加入财报、交易统计、二级股票池、walk-forward 验证和模型集成。
