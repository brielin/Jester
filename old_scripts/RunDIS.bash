#!/bin/bash
d=$1
export OMP_NUM_THREADS=1
python -m jester -v --test --file ~/AI/data/plink/$d/bed/${d}_h2c.ALL --pheno ~/AI/data/plink/${d}/bed/${d}_58C.pheno --out ~/AI/PhanHerit/res/${d}_58C_L0
python -m jester -v --test --file ~/AI/data/plink/$d/bed/${d}_h2c.ALL --pheno ~/AI/data/plink/${d}/bed/${d}_NBS.pheno --out ~/AI/PhanHerit/res/${d}_NBS_L0
