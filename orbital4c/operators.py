import numpy as np
import numpy.linalg as LA
from orbital4c import complex_fcn as cf
from orbital4c import orbital     as orb
from vampyr    import vampyr3d    as vp


class SpinorbGenerator():

    def __init__(self, mra, guessorb, c, origin, prec):
        self.prec   = prec
        self.mra = mra 
        self.guessorb = guessorb
        self.c = c
        self.origin = origin
        self.complexfc = None

        if self.guessorb == 'slater':
            print('cazzo')
        elif guessorb == 'gaussian':
################################   DEFINE GAUSSIAN FUNCTION AS GUESS  ################################
            a_coeff = 3.0
            b_coeff = np.sqrt(a_coeff/np.pi)**3
            gauss = vp.GaussFunc(b_coeff, a_coeff, self.origin)
            gauss_tree = vp.FunctionTree(self.mra)
            vp.advanced.build_grid(out=gauss_tree, inp=gauss)
            vp.advanced.project(prec=self.prec, out=gauss_tree, inp=gauss)
            gauss_tree.normalize()
#################################### DEFINE ORBITALS (C FUNCTION) ####################################
            orb.orbital4c.mra = self.mra
            orb.orbital4c.light_speed = self.c
            cf.complex_fcn.mra = self.mra
            self.complexfc = cf.complex_fcn()
            self.complexfc.copy_fcns(real=gauss_tree)


    def __call__(self, component):
        phi = orb.orbital4c()
        if component == 'La':
            phi.copy_components(La=self.complexfc)
        elif component == 'Lb':
            phi.copy_components(Lb=self.complexfc)
        else:
            'Invalid component'
        phi.init_small_components(self.prec/10)
        phi.normalize()
        return phi

class Operator():
    def __init__(self, mra, prec, Psi):
        self.mra = mra
        self.prec = prec
        self.Psi = Psi

    def matrix(self):
        n_orbitals = len(self.Psi)
        mat = np.zeros((n_orbitals, n_orbitals))
        for i in range(n_orbitals):
            si = self.Psi[i]
            Osi = self(si)
            for j in range(i+1):
                sj = self.Psi[j]
                val = sj.dot(Osi)
                print(i,j,val)
#                mat[j][i] = sj.dot(Osi)
                if (i != j):
                    mat[i][j] = np.conjugate(mat[j][i]) 
        return mat

class CoulombDirectOperator(Operator):
    def __init__(self, mra, prec, Psi):
        super().__init__(mra, prec, Psi)
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.potential = None
        self.setup()

    def setup(self):
        rho = vp.FunctionTree(self.mra)
        rho.setZero()
        for i in range(0, len(self.Psi)):
            rho += self.Psi[i].density(self.prec)
        rho.crop(self.prec)
        self.potential = (4.0*np.pi)*self.poisson(rho).crop(self.prec)

    def __call__(self, phi):
        return orb.apply_potential(1.0, self.potential, phi, self.prec)

class CoulombExchangeOperator(Operator):
    def __init__(self, mra, prec, Psi):
        super().__init__(mra, prec, Psi)
        self.poisson = vp.PoissonOperator(mra=self.mra, prec=self.prec)
        self.potential = None

    def __call__(self, phi):
        output = orbital4c()
        for i in range(0, len(self.Psi)):
            V_ij = complex_fcn()
            overlap_density = self.Psi[i].exchange(phi, self.prec)
            V_ji.real = self.poisson(overlap_density.real)
            V_ji.imag = self.poisson(overlap_density.imag)
            tmp = orb.apply_complex_potential(1.0, V_ij, Psi[i])
            output += temp                 
        output *= 4.0 * np.pi
        output.crop(prec)
        return output

class FockOperator(Operator):
    def __init__(self, mra, prec, Psi, operators, factors, der = "ABGV"):
        super().__init__(mra, prec, Psi)
        self.der = der
        self.operators = operators
        self.factors = factors

    def __call__(self, phi):
        Fphi = orb.apply_dirac_hamiltonian(phi, self.prec, self.der)
        for i in range(len(self.operators)):
            Fphi += self.factors[i] * self.operators[i].phi
        Fphi.crop(prec)
        return Fphi

