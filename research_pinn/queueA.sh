#!/bin/bash
# usage: queueA.sh gpu  (reads jobs from stdin)
cd /home/cpozogonzalez/saetass/research_pinn; . ../.venv/bin/activate
while read -r job; do CUDA_VISIBLE_DEVICES=$1 python weakA.py $job > logs/A_$(echo $job | tr ' .' '__').log 2>&1; done
