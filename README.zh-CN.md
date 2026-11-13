# 新加坡空气质量预测

[English](README.md) | 简体中文

本项目围绕 AI for Social Good（用 AI 造福社会）主题，研究空气污染预测。

## 目标

使用最近 24–48 小时 PM2.5 观测数据构成滑动窗口，预测新加坡未来 1–6 小时每个小时的污染浓度，帮助人们规划出行和户外活动。

## 结果概览

东部区域，测试期 2024-01 至 2026-09（建模过程中的任何选择都没有使用这段数据）：

| 模型 | 平均 RMSE（1–6 小时） | 平均 MAE（1–6 小时） | 污染升高时段平均 RMSE（≥ 56 µg/m³） |
| --- | --- | --- | --- |
| Persistence 基线（“保持不变”） | 6.51 | 4.63 | **24.0** |
| “昨天同一时刻”基线 | 9.28 | 6.30 | 47.6 |
| 岭回归 | 5.65 | 4.02 | 26.2 |
| 随机森林 | 5.62 | 4.02 | 25.1 |
| **梯度提升（按验证集选出）** | **5.63** | **4.01** | 25.9 |

- 所选模型的 RMSE 比 Persistence 基线在 1 小时预测上低 **7%**，在 6 小时预测上低 **16%**，五个区域全部优于基线。
- **主要不足：** 雾霾期间（污染升高时段）会低估浓度，略差于 Persistence 基线。仅凭过去的 PM2.5 无法预见雾霾的突然爆发。详见 `notebooks/02_modelling.ipynb` 第 4 节。

所有数字都来自 `results/`，可以用下文的命令重新生成。

## 仓库结构

```
data/
  singapore-pm25-hourly-raw.csv   NEA 原始数据（未改动）
  README.md                       数据来源、字段和质量检查
src/
  data.py        读取原始 CSV，修复 4 个写错的时间戳
  features.py    滞后、滚动统计和日历特征；按时间划分数据集
  models.py      基线、候选模型和超参数网格
  metrics.py     按预测步长计算 RMSE / MAE，并单独计算污染升高时段
  train.py       端到端流程：调参、测试、选出最佳模型、各区域审查
  evaluate.py    评估图表和最大误差表
  predict.py     批量预测演示：预测未来 1–6 小时
  eda.py         数据探索 Notebook 使用的画图函数
notebooks/
  01_data_exploration.ipynb   数据长什么样，以及它如何影响建模
  02_modelling.ipynb          模型比较、评估和失败案例
reports/figures/  所有图表（可用下文命令重新生成）
results/          指标表和汇总（由 src.train 生成）
tests/            自动化测试（pytest）
```

## 安装

需要 Python 3.12。开发环境为 macOS、Apple M 系列 CPU（10 核），不需要 GPU。

```bash
git clone https://github.com/ricardoypeng/singapore-air-quality-forecasting.git
cd singapore-air-quality-forecasting
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

数据集已包含在 `data/` 中，无需另外下载。

## 复现全部结果

在项目文件夹中、激活环境后运行。所有随机种子均已固定（`random_state=42`）。

| 步骤 | 命令 | 输出 | 耗时* |
| --- | --- | --- | --- |
| 1. 清洗数据（可选） | `python -m src.data` | `data/processed/pm25_hourly_clean.csv` | 几秒 |
| 2. 训练并比较模型 | `python -m src.train` | `results/`、`models/best_model.joblib` | 约 2 分钟 |
| 3. 评估图表 | `python -m src.evaluate` | `reports/figures/07_*` 至 `13_*`、`results/worst_errors.csv` | 约 10 秒 |
| 4. 数据探索图表 | 运行 `notebooks/01_data_exploration.ipynb` | `reports/figures/01_*` 至 `06_*` | 几秒 |
| 5. 预测演示 | `python -m src.predict` | 未来 1–6 小时的预测 | < 1 秒 |
| 测试 | `pytest` | 全部通过 | 约 15 秒 |

\*在 10 核笔记本电脑上测得。每个步骤都会自行清洗原始数据，第 1 步只在想查看清洗后的文件时才需要。`python -m src.train --quick` 使用极小的参数网格做快速检查。

`models/best_model.joblib` 不存放在仓库中，由第 2 步重新生成。

### Docker（可选）

```bash
docker build -t sg-pm25 .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/models:/app/models" sg-pm25
docker run --rm -v "$PWD/models:/app/models" sg-pm25 python -m src.predict
```

## 方法

1. **清洗**（`src/data.py`）：原始文件其实是按时间顺序排列、连续不断的小时序列，但有 4 行时间戳录入错误，恰好位于看似“缺失”的那个小时。程序先检查前后两行，再按位置重新标注。
2. **特征**（`src/features.py`）：对预测起点 *t*，使用目标区域过去 24 小时的值、其他四个区域过去 3 小时的值、6 小时和 24 小时的滚动均值 / 标准差 / 最大值、1 小时变化量、全岛均值，以及小时、月份（正弦 / 余弦编码）和星期几，共 49 个特征。目标为 *t*+1 … *t*+6 的值。特征只使用 *t* 及之前的数据。
3. **按时间划分**（从不打乱），各部分之间留 6 小时间隔：训练集 2014-04 至 2022，验证集 2023，测试集 2024 至 2026-09。
4. **模型**（`src/models.py`），不使用深度学习：两个基线、岭回归、随机森林和直方图梯度提升，分别尝试是否对目标取对数（共 32 组设置）。超参数在验证年上选择；最终模型在训练集 + 验证集上重新训练，只在测试集上评估一次。最佳模型按验证集成绩选出。
5. **评估**（`src/metrics.py`、`src/evaluate.py`）：按预测步长计算全部时段和污染升高时段（≥ 56 µg/m³，NEA Band II）的 RMSE 与 MAE，按污染水平分析偏差，按区域审查，计算置换特征重要性，并列出最大误差。

## 建议与局限

- 适合日常出行规划（例如“接下来几个小时是否保持在正常范围？”），明显优于“假设不变”。
- 不应作为唯一的雾霾预警：它会低估高污染。下一步可以加入风向、火点数量和区域预报，或在训练时提高雾霾时段的权重。
- 测试期内只有 2026 年出现了明显雾霾，污染升高时段的指标只基于每个步长约 144 个小时。
- 各区域数值是 NEA 的区域平均值，不代表街道层面的实际暴露。

## 预测演示

```bash
python -m src.predict                          # 使用 NEA 数据最后 48 小时
python -m src.predict --input recent.csv       # 使用你自己的读数
```

输入 CSV 需要一个 `timestamp` 列和五个区域列，并至少包含 24 个连续小时。如果部署上线，应监控预测延迟，以及预测值与之后到达的实际读数之间的滚动误差；误差上升或输入超出训练范围，说明出现数据漂移，需要重新训练。

## 协作约定

请先 Fork 本仓库，在自己的 Fork 中创建 `dev` 分支进行开发，再通过 Pull Request（PR）向本仓库的 `main` 分支提交修改。尽量将 AI 生成的指令或约束文件（例如 `AGENTS.md`、`CLAUDE.md`）保留在本地，不提交到仓库中。

## 数据许可

包含来自新加坡国家环境局（NEA）的数据，经 data.gov.sg 获取，依据 [Singapore Open Data Licence](https://data.gov.sg/open-data-licence) 使用。
