REM ============================================================
REM  IWSLT2017 英→德 小规模 Seq2Seq Transformer 训练脚本 (10K数据)
REM  兼容 Windows PowerShell / PyCharm 终端
REM ============================================================

REM 基线模型（8头注意力）
python train_iwslt_10k.py --train_size 10000 --val_size 2000 --d_model 256 --d_ff 512 --n_heads 8 --n_layers 2 --batch_size 32 --epochs 8 --lr 3e-4 --run_name 10k_baseline

REM 消融 1：去掉位置编码
python train_iwslt_10k.py --train_size 10000 --val_size 2000 --d_model 256 --d_ff 512 --n_heads 8 --n_layers 2 --batch_size 32 --epochs 8 --lr 3e-4 --no_pos_enc --run_name 10k_ablate_no_pos

REM 消融 2：减少层数
python train_iwslt_10k.py --train_size 10000 --val_size 2000 --d_model 256 --d_ff 512 --n_heads 8 --n_layers 1 --batch_size 32 --epochs 8 --lr 3e-4 --run_name 10k_ablate_layers1

REM 消融 3：减小 FFN 宽度
python train_iwslt_10k.py --train_size 10000 --val_size 2000 --d_model 256 --d_ff 256 --n_heads 8 --n_layers 2 --batch_size 32 --epochs 8 --lr 3e-4 --run_name 10k_ablate_ff256
