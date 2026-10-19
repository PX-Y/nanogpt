运行 `train_xl_periodic_v6_4bit_15k_lr1e4.py` 
命令行可以是直接python 也可以是 `python -u train_xl_periodic_v6_4bit_15k_lr1e4.py \
  --lr_decay_iters=25000 --hybrid_end_iter=14000 \
  --v3_wave_starts=14000:14250:14500 --v3_wave_deadlines=14250:14500:14750 \
  --v3_wave_max_extensions=0` 
  前者平均loss更低一点  后者final loss低
