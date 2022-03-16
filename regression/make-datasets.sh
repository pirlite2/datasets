#!/bin/bash

# Generate data files -- pir -- 16.3.2022

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
					./format-$j-data.py
				else
					chmod 700 format-aquatic-toxicity-data.py
					./format-aquatic-toxicity-data.py
				fi
			cd ..
		fi

	done


