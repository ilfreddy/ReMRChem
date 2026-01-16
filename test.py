########## Define Enviroment #################
from orbital4c import complex_fcn as cf
from orbital4c import orbital as orb
from orbital4c import nuclear_potential as nucpot
from orbital4c import r3m as r3m
from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1

import argparse
import numpy as np
import numpy.linalg as LA
import sys, getopt

import one_electron as oneel
import two_electron as twoel
import starting_guess as sg

import importlib
importlib.reload(orb)

import fileinput
import sys

# read the input
input_blob = ""
print("Input file:", sys.argv[1])
output_file = "outputs/output" + sys.argv[1].replace("inputs/input", "")
print("Output file:", output_file)
for line in fileinput.input():
    input_blob += line


# Clear the output file before writing anything
with open(output_file, "w") as f:
    pass

oneel.write_and_print(output_file,"**************************************")

oneel.write_and_print(output_file, "INPUT FILE CONTENT:")
oneel.write_and_print(output_file,input_blob)
oneel.write_and_print(output_file,"**************************************")
print()
exec(input_blob)

#
# 1. This code works now only for atoms and up to two electrons with KTRS
# 2. Orbital guess obtained by using NR hydrogenionic 1s orbital
# 3. The input file is a Python code mostly containing variable allocation
# 4. Input parsing is executing that python input after reading it
# 5. Nuclear potential selected manually (Gaussian now to reproduce Harrison's results)
#


if (auto_box):
    box = int(np.ceil(float(50/molecule[0][1])))

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-box, box], order=order, max_depth=25)
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
charge = molecule[0][1]
position = [molecule[0][2],molecule[0][3],molecule[0][4]]
radius = molecule[0][5]
epsilon = molecule[0][6]



################### Define V potential ######################
Peps = vp.ScalingProjector(mra, prec/10)
V_tree = vp.FunctionTree(mra)

if(computePotential):
    typenuc = potential
    f = 0
    if(potential == "gaussian"):
        print("Gaussian potential")
        f = lambda x: nucpot.gaussian_potential(x, position, charge, epsilon)
        V_tree = Peps(f)
    elif(potential == "coulomb_HFYGB"):
        print("Harrison potential")
        f = lambda x: nucpot.coulomb_HFYGB(x, position, charge, prec/10)
        V_tree = Peps(f)
    elif(potential == "point_charge"):
        print("point charge potential")
        f = lambda x: nucpot.point_charge(x, position, charge)
        V_tree = Peps(f)
    elif(potential == "fermi_dirac"):
        print("Fermi Dirac potential")
        if radius == 0:
            with open("Half_Charge_Radius.txt", "r") as f:
                half_charge_radius_dict = {}
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        symbol, value = parts
                        if symbol == molecule[0][0]:
                            HCR = float(value)
                            break
        else:
            HCR = radius

        print(f"-> Using Half Charge Radius for {molecule[0][0]}: {HCR}")
            
        V_tree = nucpot.Fermi_Dirac(position, charge, box, mra, order, prec, HCR)
    else:
        exit(-1)
    #V_tree = Peps(f)
elif(readPotential):
    V_tree.loadTree(f"potential")

if(savePotential):
    V_tree.saveTree(f"potential")

# showing the variables used
print()
print("------------------------------------")
print("      Calculation parameters ")
print("------------------------------------")
print("light_speed =", light_speed)
print("derivative =", derivative)
print("Nuclear Potential Type =", potential)
print("box =", box)
print("precision =", prec)
print("order =", order)
print("threshold =", thr)
print("Charge =", charge)
        
print("Number of Atoms = ", len(molecule))
print(molecule)
print()
print()
#############################START WITH CALCULATION###################################
spinorb1 = orb.orbital4c()
spinorb2 = orb.orbital4c()
if readOrbitals:
    orbitalName = "spinorb1"
    spinorb1.read(orbitalName)
else:
    spinorb1 = sg.make_NR_starting_guess(position, charge, mra, prec)
spinorb2 = spinorb1.ktrs(prec)

if saveGuess:
    spinorb1.save("guess1")

run_D_1e       = scf and not D2 and not two_electrons
run_D2_1e      = scf and     D2 and not two_electrons
run_D_2e       = scf and not D2 and     two_electrons and not ktrs
run_D2_2e      = scf and     D2 and     two_electrons and not ktrs
run_D_2e_ktrs  = scf and not D2 and     two_electrons and     ktrs
run_D2_2e_ktrs = scf and     D2 and     two_electrons and     ktrs



if run_D_1e:
    spinorb1 = oneel.gs_D_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge, output_file)

if run_D2_1e:
    spinorb1 = oneel.gs_D2_1e(spinorb1, V_tree, mra, prec, thr, derivative, charge, output_file)

if run_D_2e:
    print("NOT PROPERLY TESTED")
    exit(-1)
    spinorb1, spinorb2 = twoel.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec, derivative)

if run_D2_2e:
    print("NOT PROPERLY TESTED")
    exit(-1)
    spinorb1, spinorb2 = twoel.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, derivative)

if run_D_2e_ktrs:
    spinorb1, spinorb2 = twoel.coulomb_gs_2e(spinorb1, V_tree, mra, prec, thr, derivative, output_file)

if run_D2_2e_ktrs:
    spinorb1, spinorb2 = twoel.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, thr, derivative, output_file)

if runGaunt:
    twoel.calcGauntPert(spinorb1, spinorb2, mra, prec)

if runGaugeA:
    twoel.calcGaugePertA(spinorb1, spinorb2, mra, prec)

if runGaugeB:
    twoel.calcGaugePertB(spinorb1, spinorb2, mra, prec)

if runGaugeC:
    twoel.calcGaugePertC(spinorb1, spinorb2, mra, prec)

if runGaugeD:
    twoel.calcGaugePertD(spinorb1, spinorb2, mra, prec)

if runGaugeDelta:
    twoel.calcGaugeDelta(spinorb1, spinorb2, mra, prec)

if saveOrbitals:
    spinorb1.save("spinorb1")



oneel.write_and_print(output_file, "")

oneel.write_and_print(output_file, "PARAMETERS:")
oneel.write_and_print(output_file, f"molecule    = {molecule}")
oneel.write_and_print(output_file, f"D2          = {D2}")
oneel.write_and_print(output_file, f"prec        = {-int(np.log10(prec))}")
oneel.write_and_print(output_file, f"order       = {order}")
oneel.write_and_print(output_file, f"derivative  = {derivative}")
oneel.write_and_print(output_file, f"box         = {box}")
if position == [0.0, 0.0, 0.0]:
    centerd = True
else:
    centerd = False
oneel.write_and_print(output_file, f"centered    = {centerd}")

oneel.write_and_print(output_file, "")
oneel.write_and_print(output_file, "-> ID calculation ")
oneel.write_and_print(output_file, "-------------------------------")
oneel.write_and_print(output_file, f"{molecule[0][0]} {int(centerd)} {int(D2)} {-int(np.log10(prec))} {derivative} {box}")
oneel.write_and_print(output_file, "-------------------------------")
