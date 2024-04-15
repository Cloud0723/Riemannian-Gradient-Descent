set -x
set -e
epoch=5
export WANDB_MODE=offline
for ((i=0; i<epoch; i++))
do
    python cart-pole-Ra2c.py --algoname Riemannian --m 16 --r 4 --n 4 --seed ${i}
    python cart-pole-Ra2c.py --algoname nonRiemannian --m 16 --r 4 --n 4 --seed ${i}
done