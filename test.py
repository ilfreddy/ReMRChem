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

import one_electron
import two_electron

import importlib
importlib.reload(orb)

import fileinput

def make_starting_guess():
    gauss_tree_tot = vp.FunctionTree(mra)
    gauss_tree_tot.setZero()
    a_coeff = 3.0
    b_coeff = np.sqrt(a_coeff/np.pi)**3
    AO_list = []
    for atom in coordinates:
        gauss = vp.GaussFunc(b_coeff, a_coeff, [atom[2], atom[3], atom[4]])
        gauss_tree = vp.FunctionTree(mra)
        vp.advanced.build_grid(out=gauss_tree, inp=gauss)
        vp.advanced.project(prec=prec, out=gauss_tree, inp=gauss)
        AO_list.append(gauss_tree)
    gauss_tree_sum = vp.sum(AO_list)

    La_comp = cf.complex_fcn()
    La_comp.copy_fcns(real = gauss_tree_sum)
    spinorb1.copy_components(La = La_comp)
    spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    spinorb1.cropLargeSmall(prec)
    spinorb2 = spinorb1.ktrs(prec)
    return spinorb1, spinorb2

input_blob = ""
for line in fileinput.input():
    input_blob += line
exec(input_blob)

################# Call MRA #######################
mra = vp.MultiResolutionAnalysis(box=[-box, box], order=order, max_depth = 30)
orb.orbital4c.mra = mra
orb.orbital4c.light_speed = light_speed
cf.complex_fcn.mra = mra
coordinates = molecule

################### Define V potential ######################
V_tree = vp.FunctionTree(mra)

if(computePotential):
    Peps = vp.ScalingProjector(mra, prec/10)
    typenuc = potential
    f = lambda x: nucpot.nuclear_potential(x, coordinates, typenuc, mra, prec, derivative)
    V_tree = Peps(f)
    print("Define V", potential, "DONE")
    com_coordinates = nucpot.calculate_center_of_mass(coordinates)

if(readPotential):
    V_tree.loadTree(f"potential")

if(savePotential):
    V_tree.saveTree(f"potential")
        
print("Number of Atoms = ", len(molecule))
print(coordinates)

#############################START WITH CALCULATION###################################
spinorb1 = orb.orbital4c()
spinorb2 = orb.orbital4c()

if readOrbitals:
    spinorb1.read("spinorb1")
    spinorb2 = spinorb1.ktrs(prec)
else:
    spinorb1, spinorb2 = make_starting_guess()

run_D_1e       = scf and not D2 and not two_electrons
run_D2_1e      = scf and     D2 and not two_electrons
run_D_2e       = scf and not D2 and     two_electrons and not ktrs
run_D2_2e      = scf and     D2 and     two_electrons and not ktrs
run_D_2e_ktrs  = scf and not D2 and     two_electrons and     ktrs
run_D2_2e_ktrs = scf and     D2 and     two_electrons and     ktrs


length = 2 * box
print("Using derivative ", derivative)

if run_D_1e:
    spinorb1 = one_electron.gs_D_1e(spinorb1, V_tree, mra, prec, thr, derivative)

if run_D2_1e:
    spinorb1 = one_electron.gs_D2_1e(spinorb1, V_tree, mra, prec, thr, derivative)

if run_D_2e:
    print("NOT PROPERLY TESTED")
    exit(-1)
    spinorb1, spinorb2 = two_electron.coulomb_gs_gen([spinorb1, spinorb2], V_tree, mra, prec, derivative)

if run_D2_2e:
    print("NOT PROPERLY TESTED")
    exit(-1)
    spinorb1, spinorb2 = two_electron.coulomb_2e_D2([spinorb1, spinorb2], V_tree, mra, prec, derivative)

if run_D_2e_ktrs:
    spinorb1, spinorb2 = two_electron.coulomb_gs_2e(spinorb1, V_tree, mra, prec, thr, derivative)

if run_D2_2e_ktrs:
    spinorb1, spinorb2 = two_electron.coulomb_2e_D2_J([spinorb1, spinorb2], V_tree, mra, prec, thr, derivative)

if runGaunt:
    two_electron.calcGauntPert(spinorb1, spinorb2, mra, prec)

if runGaugeA:
    two_electron.calcGaugePertA(spinorb1, spinorb2, mra, prec)

if runGaugeB:
    two_electron.calcGaugePertB(spinorb1, spinorb2, mra, prec)

if runGaugeC:
    two_electron.calcGaugePertC(spinorb1, spinorb2, mra, prec)

if runGaugeD:
    two_electron.calcGaugePertD(spinorb1, spinorb2, mra, prec)

if runGaugeDelta:
    two_electron.calcGaugeDelta(spinorb1, spinorb2, mra, prec)

if saveOrbitals:
    spinorb1.save("spinorb1")

