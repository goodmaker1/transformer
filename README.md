# 📘 项目运行说明

## 一、硬件与环境要求
- **操作系统**：Ubuntu 20.04 / Windows 10 及以上  
- **Python 版本**：≥3.9  
- **GPU**：推荐使用 NVIDIA GPU（显存 ≥8GB）以加速训练  
- **CUDA/cuDNN**：CUDA ≥11.3，对应 cuDNN ≥8.2  
- **显卡驱动**：≥460.xx  

---

## 二、环境配置
```bash
# 克隆项目
git clone https://github.com/yourname/yourproject.git
cd yourproject

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate     # Linux
venv\Scripts\activate        # Windows

# 安装依赖
pip install -r requirements.txt
```

## 三、项目结构
```bash
├── src/                    # 主代码目录（模型、数据、训练脚本等）
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── scripts/
│   └── run.sh              # 训练运行脚本（可一键启动所有实验）
│
├── results/                # 放置训练曲线图与结果表格
│   ├── loss_curve.png
│   └── metrics_table.csv
│
├── requirements.txt
└── README.md
```

## 四、运行命令说明
```bash
bash scripts/run.sh
```
