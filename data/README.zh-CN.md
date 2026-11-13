# 原始数据

[English](README.md) | 简体中文

## 包含的来源

- [NEA五区域每小时PM2.5原始CSV](singapore-pm25-hourly-raw.csv)：109,584行，2014-04-01 01:00至2026-10-01 00:00，新加坡本地时间。首列是时间戳，其余为north/south/east/west/central，单位µg/m³。原始文件不修改。
- [S107气象来源清单](weather_raw_s107/sources.json)：五个变量×2017–2024八年，共40份按East Coast Parkway站点S107筛选的观测CSV，用gzip无损压缩。保留观测时间、站点、数值、单位与发布时间，未做小时聚合或插补。
- [SHA256校验清单](raw_sources.json)：PM CSV与40份气象观测的校验值，以及应生成的窗口数量。

PM来源：[NEA Historical 1-hr PM2.5](https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view)。气象来源集合：[风速](https://data.gov.sg/collections/2280/view)、[风向](https://data.gov.sg/collections/2281/view)、[雨量](https://data.gov.sg/collections/2279/view)、[温度](https://data.gov.sg/collections/2246/view)、[湿度](https://data.gov.sg/collections/2278/view)。来源清单保留每份气象文件的dataset ID、网址、站点筛选条件、大小与哈希。数据由NEA发布、通过data.gov.sg取得，使用遵循[Singapore Open Data Licence](https://data.gov.sg/open-data-licence)。

## 准备规则

[PM准备程序](../source/prepare_data.py)对时间格式异常、缺小时、重复小时或浓度无效的日期整天剔除，不修复时间戳、不插值、不填零。其临时划分在气象合并时弃用，只复用观测。

[气象准备程序](../source/prepare_weather_data.py)对特征时刻t使用[t−2h,t−1h)的历史观测，并检查发布时间不晚于t。温湿度每小时至少45个不同分钟且连续缺口不超过15分钟；风速/风向要求同样充分的配对分钟，knots转m/s后计算u/v。雨量须有12个五分钟时间片，修正恰好一秒的采样刻度抖动，缺测不当成无雨。

联合请求区间2017–2024。一天任一小时缺少PM或有效气象就整天剔除，剩余1,753天按floor(0.8N)、floor(0.9N)顺序切分，得到1,402/175/176天。输入可用分区边界前已知历史，六个目标必须全部属于自身分区；窗口不跨中断。24h→6h合法窗口数为25,668/3,331/3,818。

## 重建

在仓库根目录执行：

```bash
python -B -m source.prepare --verify-only
python -B -m source.prepare
python -B main.py --check
```

[准备入口](../source/prepare.py)读取仓库内的原始观测，生成小时数据和合法训练窗口。训练命令见[项目说明](../README.zh-CN.md)。重建前先核对原始文件校验值，全程不修改原始数据。
