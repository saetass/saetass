#!/bin/bash
cd /home/cpozogonzalez/saetass/research_pinn; . ../.venv/bin/activate
export NUMBA_NUM_THREADS=4 SAETASS_MAX_THREADS=4 OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 python e18.py 256 6 1024 12 > logs/S_256.log 2>&1
CUDA_VISIBLE_DEVICES=0 python e18.py 512 8 1024 12 > logs/S_512.log 2>&1
