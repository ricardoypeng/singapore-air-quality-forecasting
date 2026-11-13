# 新加坡空气质量预测

[English](README.md) | 简体中文

从过去24小时观测直接预测东区未来1–6小时PM2.5。当前模型为HGB（六个独立时距头）和CatBoost（联合六输出树），不使用人工路由或递归反馈。东区输入186列，五区输入282列。

## 提交范围

仓库包含模型与数据准备代码、必要原始观测、依赖和说明。

- [主入口](main.py)、[训练流程](source/workflow.py)、[HGB](source/models/histogram_boosting.py)、[CatBoost](source/models/catboost_forecast.py)。
- [原始数据说明](data/README.zh-CN.md)、[原始数据校验清单](data/raw_sources.json)。
- [原始数据重建入口](source/prepare.py)、[完整实验入口](source/reproduce.py)。
- [固定训练依赖](requirements.txt)，配合Python3.10。

## 新电脑复现

原始五区域PM CSV与40份按站点筛选、gzip无损压缩的气象观测已包含，无需重新下载。数据来源和清洗规则见数据说明。

```bash
python3.10 -m venv .venv
# macOS/Linux：
source .venv/bin/activate
# Windows PowerShell则使用：
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B -m source.prepare
python -B main.py --check
python -B main.py
```

准备入口先校验41份原始观测的SHA256，再重建小时数据、顺序划分与合法窗口，核对25,668/3,331/3,818个train/validation/test窗口；此步骤不训练。PM异常日期按整天剔除，输入窗口按连续小时构造。

公共入口扫描模型目录，先完成本次发现的所有候选的validation阶段，再执行test。当前两个模型class声明八个候选，训练器不写死候选数量。预算只由训练内五折时间CV选择，边界清除跨越六小时目标；独立validation选模型，test不改选。可运行 `python -B main.py --dry-run` 查看发现的class、候选和执行阶段而不训练。原 `python -B -m source.reproduce` 入口也调用同一流程。完成的报告拒绝覆盖；另开实验用新的 `--output-dir`。

## 参考行为

既有validation选型为五区CatBoost depth6、保留500棵树，历史validation/test MSE为34.450067/18.850461；东区HGB standard选200轮，分别36.474691/18.862571。数字仅作说明中的核对基准，不提交结果文件或模型。复现实验使用Python3.10及固定版本依赖；跨平台训练数值一致性尚未验证。

## 协作

请先Fork，在自己的dev分支开发，通过PR向main提交。AI约束文件保留本地。

## 自动发现模型、抽样与出图

[模型接口说明](source/models/README.md)定义所有class遵守的统一接口。新模型只需放入模型包并声明参数及预算，无需修改公共训练器；[特征构造](source/features.py)、训练、评估和结果导出共用。

```bash
python -B main.py --list-models
python -B main.py --models hgb --scopes east
python -B main.py --sample-count 10
# 从保存结果重新抽数出图，不训练：
python -B main.py --stage report --sample-count 10
# 接口测试会动态添加临时第三模型，不拟合真实估计器：
python -B -m unittest discover -s tests -v
```

完整运行会训练全部选中候选，锁定validation选择，再测试全部候选，最后自动导出模型比较、逐时距指标、validation/test逐条预测，以及默认10组共同测试样本。样本包含已知历史、各模型特征向量、真实目标及各模型预测；按时间等距抽取起点，不按误差挑样例。

自动生成整体MSE、逐时距测试MSE、训练曲线、CV预算曲线和共同样例五类图。运行后，表与抽样数据生成在运行目录的exports子目录，图在figures子目录，checkpoint在各候选子目录。[结果导出器](source/reporting.py)可单独重跑，不构造模型、不调用fit或predict。

需要分阶段运行时，先对本次全部选中候选执行 `--stage validation --config <候选名>`，再执行 `--stage test`，各次使用一致运行选项。全部validation完成前禁止测试；`--stage report` 只读取保存结果。已有validation/test拒绝覆盖；代码、候选集合或运行设置变化后使用新输出目录。
