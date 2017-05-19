#!/bin/bash
### Clean up working directory.
mv -v *.png plots
mv -v *.out *.err logs
#
#
#BSUB -n 1                          # number of tasks in job
#BSUB -o %J.out                     # output file name in which %J is replaced by the job ID
#BSUB -e %J.err                     # error file name in which %J is replaced by the job ID
#BSUB -a mympi
###BSUB -R "span[ptile=1]"          # tiling.
###BSUB -q psfehhiprioq             # queue for live data monitoring while experiment is alive.
#BSUB -q psanaq                     # queue  for offline data analysis.

### Run CXIMonitor on experiment 40112 run 32 in non-interactive mode no background subtraction.
python CXIMonitor.py --experiment k8816 --run 32 --prefix $LSB_JOBID
#python CXIMonitor.py --experiment 40112 --run 32 --events 23,32 --prefix $LSB_JOBID
