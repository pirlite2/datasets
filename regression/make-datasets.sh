#!/bin/bash

# Generate data files -- pir -- 16.3.2022
# Modified to use Python variable -- pir -- 10.5.2022

PYTHON=python3

for i in $(ls -d */)
	do
	j=${i%%/}
	#cd $j

	if [[ $j != "check-binary-dat-file" ]]
		then
			echo processing $j
			cd $j
			if [[ $j != "qsar_aquatic_toxicity" ]]
				then
					chmod 700 format-$j-data.py
					$PYTHON format-$j-data.py
				else
					chmod 700 format-aquatic-toxicity-data.py
					$PYTHON format-aquatic-toxicity-data.py
				fi
			cd ..
		fi

	done


