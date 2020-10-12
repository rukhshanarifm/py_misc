'''
I utilized the following resource(s) for this dofile:

https://stackoverflow.com/questions/21263668/run-stata-do-file-from-python

BEFORE RUNNING: make sure subprocess is installed
'''
import subprocess

dofile = "ENTER DOFILE NAME (FORMAT: test_file.do)"

'''
Here, you will need to add the path to the Stata Application. 
For example: If I am using Stata 13 and its MP version, the 
path to the Stata application might be /Applications/Stata 13/StataMP.app
'''
cmd = ["PATH TO STATA APPLICATION", "do", dofile]

## Run do-file
subprocess.call(cmd)
