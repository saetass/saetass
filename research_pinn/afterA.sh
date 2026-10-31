#!/bin/bash
cd /home/cpozogonzalez/saetass/research_pinn
while pgrep -f "queueA.sh" >/dev/null; do sleep 30; done
printf "energylog 1 270 6.4 3.5 0 12000 fair\nvpinnlog 1 270 6.4 3.5 0 12000 fair\nstronglog 1 270 6.4 3.5 0 12000 fair\nsglog 1 270 6.4 3.5 0 12000 fair\n" > jobF.txt
./queueA.sh 1 < jobF.txt
