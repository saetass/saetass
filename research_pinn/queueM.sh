#!/bin/bash
cd /home/cpozogonzalez/saetass/research_pinn; . ../.venv/bin/activate
while read -r job; do CUDA_VISIBLE_DEVICES=$1 python weakM.py $job > logs/M_$(echo $job | tr ' .' '__').log 2>&1; done
