from scipy.constants import hbar
from scipy.linalg import eig, inv
from scipy.special import legendre, laguerre, erf, gamma
from scipy.special import gamma
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1
from orbital4c import orbital as orb


import numpy as np

def calculate_center_of_mass(atoms_list):
    total_mass = 0.0
    center_of_mass = [0.0, 0.0, 0.0]

    for atom in atoms_list:
        # Assuming each atom has mass 1.0 (modify if necessary)
        mass = 1.0
        total_mass += mass

        # Update the center of mass coordinates
        for i in range(3):
            center_of_mass[i] += atom[i+2] * mass

    # Calculate the weighted average to get the center of mass
    for i in range(3):
        center_of_mass[i] /= total_mass

    return center_of_mass

def nuclear_potential(position, atoms_list, typenuc, mra, prec, der):
    potential = 0
    for atom in atoms_list:
        charge = atom[1]
        atomname = atom[0]
        atom_coordinates = [atom[2], atom[3], atom[4]]
        if typenuc == 'point_charge':
            atomic_potential = point_charge(position, atom_coordinates, charge)
        elif typenuc == 'coulomb_HFYGB':
            atomic_potential = coulomb_HFYGB(position, atom_coordinates, charge, prec)
        elif typenuc == 'gaussian':
            atomic_potential = gaussian_potential(position, atom_coordinates, charge, atomname)
        else:
            print("Potential not defined")
            exit(-1)
        potential += atomic_potential
    return potential

def point_charge(position, center , charge):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    return charge / distance

def smoothing_HFYGB(charge, prec):
    factor = 0.00435 * prec / charge**5
    return factor**(1./3.)

def uHFYGB(r):
    u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
    return u

def coulomb_HFYGB(position, center, charge, precision):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    factor = smoothing_HFYGB(charge, precision)
    value = uHFYGB(distance/factor)
    return charge * value / factor

def get_param_homogeneous_charge_sphere(atom):
    fileObj = open("./orbital4c/param_V.txt", "r")
    RMS = ""
    for line in fileObj:
        if not line.startswith("#"):
            line = line.strip().split()
            if len(line) == 3:
               if line[0] == atom:
                   RMS = line[1]
            else:
               print("Data file not correclty formatted! Please check it!")
    fileObj.close()
    return float(RMS)

def homogeneus_charge_sphere(position, center, charge, RMS):
    RMS2 = RMS**2.0
    d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
    distance = np.sqrt(d2)
    R0 = (RMS2*(5.0/3.0))**0.5
    if distance <= R0:
          prec = charge / (2.0*R0)
          factor = 3.0 - (distance**2.0)/(R0**2.0) 
    else:
          prec = charge / distance
          factor = 1.0
    return prec * factor


def gaussian_potential(position, center, charge, epsilon):
    d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
    distance = np.sqrt(d2)
    point_charge_potential = charge / distance
    gaussian_screening = erf(np.sqrt(epsilon) * distance)
    return point_charge_potential * gaussian_screening


def Fermi_Dirac(center, charge, box_size, mra, ord, prec, C):
    
    # Define the lambda function for the Fermi-Dirac distribution
    # This parameter T is chosen to have a smooth transition and is the same for all atoms
    global T
    #T = 4.349e-5
    #T = (2.3e-5) / 0.52917721092 #recent one
    T = 0.00000989059 * np.log(81)
    #T = 2.3 / 52917.7249 # maybe grasp
    # BOOSTED BY A FACTOR OF 10 TO HAVE A SMOOTHER TRANSITION
    # Fermi Dirac distribution defined as a function of the radial distance r
    def FD(r):
        r = np.array(r)
        exponent = (r - C) / T
        prefactor = 4 * np.log(3)
        # Limit the exponent to avoid overflow
        exp_arg = np.clip(prefactor * exponent, -700, 700) # not to make it explode
        out = 1 / (1 + np.exp(exp_arg))
        return out
    


    # Define the 3D function tree to hold the charge density
    Rho_3D = vp.FunctionTree(mra)
    
    # Define a Gaussian with the inflection point in the same position of the Fermi-Dirac distribution
    beta = C**2
    beta = 1/(2*beta)
    alpha = charge * (beta/np.pi)**(3/2) # Not strictly necessay but i normalize it to have the right charge
    
    # Define the 3D Gaussian function
    gauss_3D = vp.GaussFunc(alpha=alpha, beta=beta, position=center)
    # Now I transfer the same grid of this tiny Gaussian to the Fermi-Dirac distribution so that it won't miss any feature
    vp.advanced.build_grid(out=Rho_3D, inp=gauss_3D)
    
    # Need to redefine the Fermi Dirac in 3D 
    def Fermi_Dirac_3D(position):
        d2 = ((position[0] - center[0]) ** 2 +
          (position[1] - center[1]) ** 2 +
          (position[2] - center[2]) ** 2)
        r = np.sqrt(d2)
        return FD(r) 

    # Project the 3D Fermi-Dirac distribution onto the Function Tree with the same grid as the tiny Gaussian
    vp.advanced.project(prec=prec, out=Rho_3D, inp=Fermi_Dirac_3D, abs_prec=True)
    # Integrate this function numerically in 3D to get the normalization constant
    Rho_3D = Rho_3D * 10**(12) # just because I know it is alrerady in this order of magnitude
    integral_3D = Rho_3D.integrate()



    #integral_FD_3D = FD_tree.integrate()
    #print('Integral of the Fermi-Dirac distribution via radial integration:', integral_FD)
    print('Integral of the Fermi-Dirac distribution via cartesian 3D integration:', integral_3D)

    RhoF_0 = charge / integral_3D
    print("The normalization constant is:", RhoF_0)
    
    Rho_3D = Rho_3D * RhoF_0
    Should_be_charge = Rho_3D.integrate()
    print('Integral of the Fermi-Dirac distribution via 3D integration:', Should_be_charge)
    if abs(Should_be_charge - charge) > prec:
        print("")
        print(">>The integral of the charge density is not correct!<<")
        exit(-1)


    # Now I get the potential by convoluting with the Poisson kernel
    P = vp.PoissonOperator(mra, prec/10)
    # Remember that by definition in atomic units the Poisson operator has a 4*pi factor
    V_tree =  np.pi*4*P(Rho_3D)
    print("3D density")
    print(Rho_3D)
    print("3D potential")
    print(V_tree)


    return V_tree


 