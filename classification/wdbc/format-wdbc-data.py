#! /usr/bin/python3

#*******************************************************************************
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#*******************************************************************************
"""
Format UCI dataset: wdbc

1) Comma separated data
2) First row entry is a patient ID
3) Class label is second column

version: 0.1
author: Peter Rockett, University of Sheffield, 14.1.2021
"""
#*******************************************************************************

import numpy as np
import random
import sys

# Parameters
filename = "wdbc.data"

noPartitions = 10
trainingPercentage = 70.0
validationPercentage = 10.0
testPercentage = 20.0
assert trainingPercentage + validationPercentage + testPercentage == 100.0, "Percentages of partitions do not sum to 100%"

#*******************************************************************************

def output_record(fDescriptor, record):
	"Output individual record to text file"
	
	recordLength = len(record)
	
	for i in range(1,recordLength):
		fDescriptor.write(str(record[i]))
		fDescriptor.write(",")
	
	tagValue = int(record[0])
	fDescriptor.write(str(tagValue))
	fDescriptor.write("\n")	
	
	return
	
#-------------------------------------------------------------------------------

def output_binary_header(fDescriptor, patternVectorLength, noExamples):
	"Write pattern vector length & no. of patterns to binary DAT file as 32-bit unsigned int"

	fDescriptor.write(np.uint32(patternVectorLength))
	fDescriptor.write(np.uint32(noExamples))

	return
	
#-------------------------------------------------------------------------------

def output_binary_record(fDescriptor, record):
	"Write individual record to binary DAT file as double & tag value as a int"
	
	recordlength = len(record)
	
	# Write pattern vector
	for i in range(1, recordLength):	# Note: range!
		fDescriptor.write(np.double(record[i]))

	# Write tag value as an int from record[1]
	fDescriptor.write(np.int32(record[0]))
		
	return

#*******************************************************************************

# Main program

fd = open(filename, "r")
if fd == None:
	print("Unable to open " + filename)
	sys.exit(1)
	
fRecoded = open(filename + "_recoded.data", "w")
if fRecoded == None:
	print("Unable to open " + filename + "_new.data")
	sys.exit(1)
	
print("Processing " + filename)

noRecords = 0
recordLength = 0
dataset = []
for line in fd:
	noRecords += 1
	
	# Tokenise input line
	line = line.rstrip("\n")
	record = line.split(",")	# Comma separated file!
	
	newRecord = []
	
	#---------------------------------------------------------------------------
	# MODIFY BELOW HERE!
	
	for i in range(1, len(record)):
		# Process class label in record[1]
		if i == 1:
			if record[1] == 'B':
				newRecord.append(0)
			else:
				# record[1] == 'M'
				newRecord.append(1)
		else:
			# Process numeric features
			newRecord.append(float(record[i]))
		
	# MODIFY ABOVE HERE!
	#---------------------------------------------------------------------------
	
	recordLength = len(newRecord)
	dataset.append(newRecord)
	output_record(fRecoded, newRecord)

fd.close()
fRecoded.close()
print(noRecords, " records formatted")
print("Recoded record length =", recordLength)

#-------------------------------------------------------------------------------

# Output disjoint training/validation/test datasets
noValidationExamples = round(validationPercentage * noRecords / 100.0)
noTestExamples = round(testPercentage * noRecords / 100)
noTrainingExamples = noRecords - noValidationExamples - noTestExamples
assert noTrainingExamples + noValidationExamples + noTestExamples == noRecords, "Datasets are not disjoint/sum to total no. of record"

print(noValidationExamples, "validation examples, ", noTestExamples, " test examples, ", noTrainingExamples, " training examples")

ranSeed = 100
for n in range(0, noPartitions):
	# CSV datasets
	fTrain = open(filename + "_training-" + str(n) + ".data", "w")
	fValidation = open(filename + "_validation-" + str(n) + ".data", "w")
	fTest = open(filename + "_test-" + str(n) + ".data", "w")
	
	# Binary .dat files
	fTrainBin = open(filename + "_training-" + str(n) + ".dat", "wb")
	fValidationBin = open(filename + "_validation-" + str(n) + ".dat", "wb")
	fTestBin = open(filename + "_test-" + str(n) + ".dat", "wb")	
	
	selectionArray = np.zeros((noRecords), dtype = bool)
	random.seed(ranSeed)
	
	# Output test dataset
	output_binary_header(fTestBin, recordLength - 1, noTestExamples)
	noTestPatternsEmitted = 0
	for i in range(0, noTestExamples):
		r = random.randint(0, noRecords - 1)
		while selectionArray[r] == True:
			r = random.randint(0, noRecords - 1)
		output_record(fTest, dataset[r])
		output_binary_record(fTestBin, dataset[r])
		selectionArray[r] = True
		noTestPatternsEmitted += 1	
	assert noTestPatternsEmitted == noTestExamples
					
	# Output validation dataset
	output_binary_header(fValidationBin, recordLength - 1, noValidationExamples)
	noValidationPatternsEmitted = 0
	for i in range(0, noValidationExamples):
		r = random.randint(0, noRecords - 1)
		while selectionArray[r] == True:
			r = random.randint(0, noRecords - 1)
		output_record(fValidation, dataset[r])
		output_binary_record(fValidationBin, dataset[r])
		selectionArray[r] = True
		noValidationPatternsEmitted += 1		
	assert noValidationPatternsEmitted == noValidationExamples
	
	# Output training dataset
	output_binary_header(fTrainBin, recordLength - 1, noTrainingExamples)
	noTrainingPatternsEmitted = 0
	for i in range(0, noRecords):
		if selectionArray[i] == False:
			noTrainingPatternsEmitted += 1	
			output_record(fTrain, dataset[i])
			output_binary_record(fTrainBin, dataset[i])
	assert noTrainingPatternsEmitted == noTrainingExamples
			
	fTrain.close()
	fValidation.close()
	fTest.close()
	
	fTrainBin.close()
	fValidationBin.close()
	fTestBin.close()
	
	ranSeed += 1
	
print(noPartitions, "training/validation/test datasets created")	

#*******************************************************************************




