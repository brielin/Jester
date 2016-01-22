#!/bin/bash
c=$1
python jester/__main__.py -v --test --file ~/AI/Replication/CAD/C4D_$c --JointTestingWindow 500 --type s --rFile ~/AI/Replication/CAD/C4D.$c.corr.out --frq ~/AI/data/1KG/phase3/C4D/EUR_SAS.chr$c.common.filtered.frq --out ~/AI/results/CAD/C4D_$c > ~/AI/results/CAD/C4D_$c.log