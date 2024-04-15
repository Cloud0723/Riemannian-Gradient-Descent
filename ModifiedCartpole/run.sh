set -x
set -e
epoch=5
export WANDB_MODE=offline
for ((i=0; i<epoch; i++))
do
    # python CartPole.py --use_riem 1 --m 8 --r 2 --n 8 --seed ${i}
    python CartPole.py --use_riem 0 --m 8 --r 2 --n 8 --seed ${i}
done