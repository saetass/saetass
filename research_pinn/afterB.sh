#!/bin/bash
cd /home/cpozogonzalez/saetass/research_pinn; . ../.venv/bin/activate
while pgrep -f "queueB.sh" >/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 python e6.py c_ss ss 16000 > logs/C_ss.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python e6.py c_be be 16000 > logs/C_be.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python e6.py c_none none 16000 > logs/C_none.log 2>&1 &
wait
