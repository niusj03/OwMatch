> ./results/cifar10/$2.log

python $1 --gpu $4 --seed 1 --name $2 \
--epochs 200 --dataset 'cifar10' --labeled-num 5 \
--batch-size 512 --labeled-ratio $3 \
--sk-epsilon 0.05 --sk-iters 10 --adjust-prior -1 \
--adathres 'OpenFree' --adathres-init 0.5 --adathres-alpha 0.99 \
--fix-thres 0.95 --queue-len 1024 --hard-logit 1 \
--has-confidence 1 --has-supervised 1 \
--ema-inf 1 --ema-alpha 0.999 \
--n-crop 4 \
>> ./results/cifar10/$2.log