# JPX Tokyo Stock Exchange Prediction Baseline

这个项目用于完成一个真实的 Kaggle 金融时间序列竞赛闭环：下载 JPX 数据、解释任务、训练 baseline、离线验证，并生成 `submission.csv`。

## 原 Kaggle 链接是什么

用户给的 `TS-3: Time series for finance` 是 Konrad Banachewicz 的教学 Notebook，不是 Kaggle Competition。它使用 NIFTY-50 股票数据讲金融时间序列常见概念：

- 收益率序列常见“波动聚集”，即高波动期和低波动期会成段出现。
- ARCH/GARCH 不是直接预测价格，而是建模残差或收益率的条件方差。
- AR-GARCH 可以把均值模型和波动率模型组合起来做预测。
- VaR 用条件均值、条件波动率和分位数估计投资组合潜在损失。

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
- Kernel version：`8`
- Submission ref：`52710807`
- Public score：`0.108`
- Private score：`0.108`
- 主要改进：线上提交脚本加入 1/2/5/10/20 日收益率、滚动收益均值、滚动波动率、调整后收盘价和成交量比等时间序列特征。

输出文件：

- `outputs/submission.csv`：本地生成的提交文件。
- `outputs/metrics.json`：验证 Sharpe、RMSE、提交文件检查。
- `outputs/feature_importance.csv`：LightGBM 特征重要性。

## 注意

这不是投资建议。该 baseline 是为了复现竞赛流程和建立可迭代起点，后续提分可以加入财报、交易统计、二级股票池、walk-forward 验证和模型集成。
