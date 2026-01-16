# ReMRChem
Python relativistic implementation

This is a Python Relativistic implementation to perform very accurate calculations 
in the MultiWavelets framework.

To run the calculations you have to create a file, an example is provided in "input.txt"
The main Python file, with which you're supposed to run calculation is called "test.py"

Make sure to have installed the vampyr library in your virtual environment.
Once this is done you can run on the terminal, in the main folder, the followinfg commang
"""
    python test.py input.txt
"""
The output will be shown in the main terminal. 

The code supportes only calculation for 1 or 2 electrons (taking advantage of the time reversal symmetry). There are 2 main possibilities for the SCF procedure: Dirac and Dirac squared Hamiltonian. Once converged, both expectation values, taken with each Hamilonian, will be computed and shown.


