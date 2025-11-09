# plot_metrics_auto_cn.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False   # 负号正常显示

NEEDED_COLS = ["epoch", "train_loss_token", "val_loss_token", "val_ppl"]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in NEEDED_COLS:
        if col not in df.columns:
            raise ValueError(f"{os.path.basename(path)} 缺少列: {col}")
    return df[NEEDED_COLS]

def moving_average(series, k: int):
    if k <= 1:
        return series
    return series.rolling(window=k, min_periods=1).mean()

def summarize(label: str, df: pd.DataFrame) -> str:
    idx = df["val_ppl"].idxmin()
    be = int(df.loc[idx, "epoch"])
    bv = float(df.loc[idx, "val_ppl"])
    fe = int(df["epoch"].iloc[-1])
    fv = float(df["val_ppl"].iloc[-1])
    return f"[{label}] 最优验证PPL={bv:.4f} @ epoch {be} | 最终PPL={fv:.4f}"

def plot_one(ycol: str, runs, out_png: str, ylabel: str, title: str):
    plt.figure()
    for label, df in runs:
        x = df["epoch"]
        y = moving_average(df[ycol], 1)
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel("训练轮数 (Epoch)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    # 把这一行改成下面这样：
    root = "../results"  # 原来是 root = "results"
    out_prefix = "../results/compare_cn"

    include_keys = ["baseline", "ablate_layers1", "ablate_ffn", "no_pos_mask"]

    files = sorted(glob.glob(os.path.join(root, "*_metrics.csv")))
    files = [f for f in files if any(k.lower() in os.path.basename(f).lower() for k in include_keys)]

    if not files:
        print("未找到匹配的 metrics 文件。")
        return

    runs = []
    for f in files:
        try:
            df = load_csv(f)
            label = os.path.splitext(os.path.basename(f))[0]
            runs.append((label, df))
        except Exception as e:
            print(f"跳过 {f}: {e}")

    print("\n====== 结果摘要 ======")
    for label, df in runs:
        print(summarize(label, df))

    plot_one("train_loss_token", runs, f"{out_prefix}_train_loss.png", "训练集损失 (Train Loss / token)", "训练损失随 Epoch 变化")
    plot_one("val_loss_token", runs, f"{out_prefix}_val_loss.png", "验证集损失 (Val Loss / token)", "验证损失随 Epoch 变化")
    plot_one("val_ppl", runs, f"{out_prefix}_val_ppl.png", "验证集困惑度 (Perplexity)", "验证集困惑度变化趋势 (越低越好)")

    print("\n已保存图像：")
    print(f" - {out_prefix}_train_loss.png")
    print(f" - {out_prefix}_val_loss.png")
    print(f" - {out_prefix}_val_ppl.png")

if __name__ == "__main__":
    main()
