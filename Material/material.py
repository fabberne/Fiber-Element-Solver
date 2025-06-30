import numpy as np
from numba import jit

class Material:
    def __init__(self):
        pass


#--------------------------------------------------------------------------------------------------------------------------------#


class Concrete(Material):
    def __init__(self):
        self.color = (0, 0, 0, 0.5)
        self.name  = "Concrete_C30_37"


    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        # Constants for the concrete material
        f_druck = 30.000  
        f_zug   =  1.280
        e_max   =  0.003
        E       =  32000

        # Constants for the stress-strain curve
        # material law from Hulail et al. (2023)
        a = min( 0.70 * f_druck ** (1/15),  1.00)
        b = max(-0.02 * f_druck **   0.8 , -0.95)
        c = 0.02 * f_druck

        # vectorized stress calculation
        stresses = np.where(strains <= 0, 
            # Negative strain (tensile behavior)
            np.clip(E * strains, -f_zug, 0),
            
            np.where(strains <= e_max, 
                # Ascending branch
                f_druck * (strains / e_max) ** ((a * (1 - strains / e_max)) /(1 + b * strains / e_max)),
                # Descending branch
                f_druck * (strains / e_max) ** (((c **(e_max / strains)) * (1 - (strains / e_max) ** c)) / (1 + (strains / e_max) ** c))
            )
        )

        # zero stress below tensile failure limit
        stresses[stresses <= -f_zug] = 0

        return stresses


    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        # Constants for the concrete material
        f_druck = 30.000  
        f_zug   =  1.280
        e_max   =  0.003
        E       =  32000

        a = min( 0.70 * f_druck ** (1/15),  1.00)
        b = max(-0.02 * f_druck **   0.8 , -0.95)
        c = 0.02 * f_druck

        # tangent modulus calculation as the derivation of the stress-strain curve
        tangents = np.empty_like(strains)
        for i in range(strains.shape[0]):
            eps = strains[i]
            if eps <= 0.0:
                # tensile: elastic up to failure, then zero
                if eps <= -f_zug / E:
                    tangents[i] = 0.0
                else:
                    tangents[i] = E
            else:
                x = eps / e_max
                if x <= 1.0:
                    # derivative of the ascending branch: σ = f_druck*x**n(x)
                    # with n(x) = a(1–x)/(1+b*x)
                    denom = 1.0 + b * x
                    n     = (a * (1.0 - x)) / denom
                    n_p   = -a * (1.0 + b) / (denom * denom)
                    # dσ/dε     = f_druck * x**n * [n'(x) * ln x + n/x] * (1/e_max)
                    tangents[i] = f_druck * x**n * (n_p * np.log(x) + n / x) / e_max
                else:
                    # derivative of the descending branch: σ = f_druck*x**m(x)
                    # with m(x) = C(x)*D(x)
                    # C = c^(1/x),  D = (1 - x^c)/(1 + x^c)
                    C   = c ** (1.0 / x)
                    D   = (1.0 - x**c) / (1.0 + x**c)
                    C_p = -np.log(c) / (x*x) * C
                    D_p = -2.0 * c * x**(c - 1.0) / (1.0 + x**c)**2
                    m   = C * D
                    m_p = C_p * D + C * D_p
                    # dσ/dε = f_druck * x**m * [m'(x) * ln x + m/x] * (1/e_max)
                    tangents[i] = f_druck * x**m * (m_p * np.log(x) + m / x) / e_max
        return tangents
        
    
#--------------------------------------------------------------------------------------------------------------------------------#


class Concrete_C30_37(Material):
    def __init__(self):
        self.color = (0, 0, 0, 0.5)
        self.name  = "Concrete_C30_37"


    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        # Constants for the bilinear concrete material model
        E       = 12000.0
        f_c     = 30.0000
        eps_c   =  0.0025
        eps_u   =  0.0250
        f_t     =  1.2800
        eps_t   = - f_t / E

        # Initialize stresses array
        stresses = np.zeros_like(strains)

        # Tension:
        tension_mask = (strains <= 0) & (strains > eps_t)
        stresses[tension_mask] = E * strains[tension_mask]

        # Compression (elastic):
        comp_elastic_mask = (strains > 0) & (strains < eps_c)
        stresses[comp_elastic_mask] = E * strains[comp_elastic_mask]

        # Compression (softening):
        comp_soften_mask = (strains >= eps_c) & (strains <= eps_u)
        stresses[comp_soften_mask] = f_c * (eps_u - strains[comp_soften_mask]) / (eps_u - eps_c)

        # Everything else (cracked tension, failed compression) remains 0
        return stresses


    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        # Constants for the bilinear concrete material model
        E       = 12000.0
        f_c     = 30.0000
        eps_c   =  0.0025
        eps_u   =  0.0250
        f_t     =  1.2800
        eps_t   = - f_t / E

        # Initialize tangent modulus array
        tangents = np.zeros_like(strains)

        # Tension:
        tension_mask = (strains <= 0) & (strains > eps_t)
        tangents[tension_mask] = E

        # Compression (elastic):
        comp_elastic_mask = (strains > 0) & (strains < eps_c)
        tangents[comp_elastic_mask] = E

        # Compression (softening):
        comp_soften_mask = (strains >= eps_c) & (strains <= eps_u)
        tangents[comp_soften_mask] = - f_c / (eps_u - eps_c)

        return tangents
        
    
#--------------------------------------------------------------------------------------------------------------------------------#


class Steel_S235(Material):
    def __init__(self):
        self.color = (0, 0, 1, 0.5)
        self.name  = "Steel_S235"


    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        # material properties for S235 steel
        E    = 210000    # N/mm2    young's modulus
        f_y  = 235       # N/mm2    yield strength
        e_y  =  f_y / E  # --       yield strain
        H    = 0.01 * E  # N/mm2    hardening modulus

        # piecewise: E or H
        return np.where(np.abs(strains) <= e_y, E, H)


    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        # material properties for S235 steel
        E    = 210000    # N/mm2    young's modulus
        f_y  = 235       # N/mm2    yield strength
        e_y  =  f_y / E  # --       yield strain
        H    = 0.01 * E  # N/mm2    hardening modulus

        stresses = np.where(
            (np.abs(strains) <= e_y),
            strains * E,
            np.sign(strains) * (f_y + H * (np.abs(strains) - e_y))
        )
        return stresses


#--------------------------------------------------------------------------------------------------------------------------------#


class Rebar_B500B(Material):
    def __init__(self):
        self.color = (0, 0.2, 1, 0.5)
        self.name  = "Rebar_B500B" 
    

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        # material properties for B500B rebar
        E   = 205000     # N/mm2    young's modulus
        f_s = 500        # N/mm2    yield strength
        e_s = f_s / E    # --       strain at ultimate compressive strength
        E_h = (1.08*f_s - f_s) / (0.05 - e_s)   # N/mm2    hardening modulus

        # piecewise: E or E_h
        return np.where(np.abs(strains) <= e_s, E, E_h)
    

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        E   = 205000     # N/mm2    young's modulus
        f_s = 500        # N/mm2    yield strength
        e_s = f_s / E    # --       strain at ultimate compressive strength
        E_h = (1.08*f_s - f_s) / (0.05 - e_s)   # N/mm2    hardening modulus

        stresses = np.where(
            (np.abs(strains) <= e_s),
            strains * E,
            np.sign(strains) * (f_s + E_h * (np.abs(strains) - e_s))
        )
        return stresses


#--------------------------------------------------------------------------------------------------------------------------------#


class Unknown(Material):

    def __init__(self):
        self.color = (1, 0, 0, 0.5)
        self.name  = "Unknown" 