#!/bin/bash
cd /home/cpozogonzalez/saetass/research_pinn; . ../.venv/bin/activate
while read -r job; do n=${job%% *}; CUDA_VISIBLE_DEVICES=$1 python e4.py $job > logs/B_$n.log 2>&1; done
