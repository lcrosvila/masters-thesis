#!/bin/bash
# for i in 2 3 4 5
for i in 6 7 8
do
	/opt/anaconda3/bin/python /home/laura/thesis/two_inputs/main.py $i "9_8_132300_reduced"
	/opt/anaconda3/bin/python /home/laura/thesis/two_inputs/main.py $i "logmel_reduced"
done
