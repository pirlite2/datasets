# Regression Datasets

All the datasets in this folder have been obtained from the UCI Repository (https://archive.ics.uci.edu/ml/datasets.php) with the exception of the dow-chemical dataset that was kindly supplied by James McDermott.

Details of the dataset preparation is given in: P.Rockett -- "Constant optimization and feature standardization in multiobjective genetic programming", Genetic Programming and Evolvable Machines, doi:10.1007/s10710-021-09410-y (2021)

To create the files in each directory, run the 'make-datasets.sh' script file in a shell.

Note: It may be necessary to change the value of the 'PYTHON' variable in the 'make-datasets.sh' script depending on how Python 3 is invoked on your local machine 

- Sometimes the default 'python' invokes version 3 of Python
- Other times you need 'python3' because 'python' invokes the legacy Python 2

If in doubt, try executing 'python --version' from the command line to see which verion of Python this invokes.


