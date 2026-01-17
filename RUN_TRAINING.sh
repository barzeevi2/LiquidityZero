#!/bin/bash
# Real training command for market making agent
# This runs a full training session with optimal parameters

source venv/bin/activate

echo "Starting real training session..."
echo "Expected duration: 2-4 hours depending on your machine"
echo ""

python -m app.agents.train \
    --timesteps 500000 \
    --n-envs 4 \
    --lr 3e-4 \
    --batch-size 64 \
    --n-steps 2048 \
    --n-epochs 10 \
    --gamma 0.99 \
    --clip-range 0.2 \
    --ent-coef 0.05 \
    --features-dim 256 \
    --days 7

echo ""
echo "Training completed! Check logs/tensorboard/ for metrics."


