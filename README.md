# 2025 年短学期课程《数据要素市场》大作业

This project [Building Marketplace for Machine Learning Data](report.pdf) by [Zhe Huang](https://github.com/Xecades), [Li Qin](https://github.com/qlsupersky) and [Pu Wang](https://github.com/V1CeVersaa) is licensed under the MIT License. Authors are listed in alphabetical order.

## 项目结构

使用 `tree . -I '.venv|__pycache__|uv.lock' --dirsfirst` 查看项目结构。

```
.
├── assets
│   ├── experiment.md     # 实验部分报告
│   ├── full_report.typ   # 完整报告
│   ├── origin_paper.pdf  # 原始论文
│   ├── paper.md          # 论文理论部分报告
│   └── result.png        # 实验结果
├── src
│   ├── market.py         # 实现整个市场，生成模拟数据
│   ├── mechanism.py      # 实现特征分配、收入函数与 Shapley 分配
│   ├── model.py          # 机器学习模型
│   ├── participants.py   # 买家和卖家
│   └── pricer.py         # 定价模块，实现价格更新
├── LICENSE
├── main.py               # 主程序
├── pyproject.toml        # 项目配置
├── README.md             # 项目说明
└── report.pdf            # 完整报告
```

## 如何运行？

```bash
uv sync
uv run main.py
```
