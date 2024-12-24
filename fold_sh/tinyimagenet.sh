> ./results/tinyimagenet/$2.log

python $1 --gpu $4 --seed 1 --name $2 \
--epochs 400 --dataset 'tinyimagenet' --labeled-num 100 \
--batch-size 256 --labeled-ratio $3 \
--sk-epsilon 0.05 --sk-iters 10 --adjust-prior -1 \
--adathres 'OpenFree' --adathres-init 0.5 --adathres-alpha 0.99 \
--fix-thres 0.6 --queue-len 1024 --hard-logit 1 \
--has-confidence 1 --has-supervised 1 \
--ema-inf 1 --ema-alpha 0.999 \
--n-crop 2 --weight_decay 1e-4 \
>> ./results/tinyimagenet/$2.log