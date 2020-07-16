"""
Convert boston housing data to CSV file

version 0.1
author: Peter Rockett, 16.7.2020
"""

import sys

fd = open("boston-housing.data", "r")
fd_out = open("boston-housing.csv", "w")

for line in fd:
    line = line.rstrip("\n")
    record = line.split(" ")

    newRecord = []
    for i in range(0, len(record)):
        if record[i] != "":
            newRecord.append(record[i])

    for i in range(0,len(newRecord)):
        fd_out.write(newRecord[i])
            
        if i < (len(newRecord) - 1):
            fd_out.write(",")
        else:
            fd_out.write("\n")
  
    print(newRecord) 
    
fd.close()
fd_out.close()
    
