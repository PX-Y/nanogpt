## File Description

- **`train_dist4_noback.py`**  
  This file follows our theoretical formulation completely.

- **`train_dist5_noback.py`**  
  Compared with dist4, the modification is made inside AdamW:  
  for parameters that were determined to be quantized in the previous step, we use **`q(x)`**;  
  for parameters that are not quantized, we still use the full-precision **`x`**.

## Training Setting

All of the above experiments are trained with **without-replacement sampling**.
