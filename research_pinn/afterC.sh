#!/bin/bash
cd /home/cpozogonzalez/saetass/research_pinn; . ../.venv/bin/activate
while pgrep -f "afterB.sh|e6.py" >/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 python e7.py d_ss ss data 12000 > logs/D_ss.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python e7.py d_none none data 12000 > logs/D_none.log 2>&1 &
wait
