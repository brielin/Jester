#!/bin/bash
export OMP_NUM_THREADS=1
python -m jester -v --test --file ~/AI/PhanHerit/data/L0/ALL.$1 --pheno ~/AI/PhanHerit/ALL.pheno --out ~/AI/PhanHerit/res/ALL_L0.$1