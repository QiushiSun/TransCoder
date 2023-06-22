#!/bin/bash

model_lst=(codet5)
#(roberta codebert graphcodebert codet5 plbart unixcoder)
# cuda=0
cd ../..
for model in "${model_lst[@]}"; do

    #crosstask
    # CUDA_VISIBLE_DEVICES=0 bash run_sample005.sh $model summarize2translate
    # CUDA_VISIBLE_DEVICES=0 bash run_sample005.sh $model cls2translate
    # CUDA_VISIBLE_DEVICES=0 bash run_sample005.sh $model translate2cls
    # CUDA_VISIBLE_DEVICES=0 bash run_sample005.sh $model summarize2cls
    # CUDA_VISIBLE_DEVICES=4 bash run_sample005.sh $model translate2summarize
    # CUDA_VISIBLE_DEVICES=5 bash run_sample005.sh $model cls2summarize
    
    # # crosslang
    # CUDA_VISIBLE_DEVICES=1 bash run_sample005.sh $model cross2ruby
    # CUDA_VISIBLE_DEVICES=1 bash run_sample005.sh $model cross2javascript
    # CUDA_VISIBLE_DEVICES=2 bash run_sample005.sh $model cross2java
    # CUDA_VISIBLE_DEVICES=2 bash run_sample005.sh $model cross2go
    # CUDA_VISIBLE_DEVICES=2 bash run_sample005.sh $model crossphp
    CUDA_VISIBLE_DEVICES=5 bash run_sample005.sh $model cross2python

done