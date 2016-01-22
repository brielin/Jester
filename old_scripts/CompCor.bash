#!/bin/bash
c=$1
python -m jester --correlate -v --file ~/AI/data/1KG/phase3/C4D/EUR.chr$c.common.filtered --JointTestingWindow 500 --out ~/AI/Replication/CAD/CG.$c