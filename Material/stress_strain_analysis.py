import numpy as np

from scipy.optimize import fsolve
from numba import jit
from collections import defaultdict

@jit(nopython=True, cache=True)
def calculate_strains_fast(eps_x, cy_values, cz_values, xsi_y, xsi_z):
    # Calculate strains based on the strain curvatures and position of the fiber
    return eps_x + cz_values * xsi_y + cy_values * xsi_z

@jit(nopython=True, cache=True)
def calculate_section_forces_fast(area_values, stresses, cy_values, cz_values):
    # Calculate section forces based on the area, stresses, and positions of the fibers
    # as a summation over all fibers
    # N: axial force, My: moment about y-axis, Mz: moment about z-axis
    N  = np.sum(area_values * stresses) / 1000
    My = np.sum(area_values * stresses * cz_values) / 1e6
    Mz = np.sum(area_values * stresses * cy_values) / 1e6
    return N, My, Mz

class stress_strain_analysis:
    def __init__(self, mesh, Nx=0, My=0, Mz=0):
        # Initialize the stress-strain analysis with a mesh and optional initial forces
        # Nx, My, Mz are the initial section forces (default to zero)
        self.mesh = mesh
        self.Nx = Nx
        self.My = My
        self.Mz = Mz

        # Initialize strain and curvature values
        self.eps_x = 0
        self.xsi_y = 0
        self.xsi_z = 0

        # Extract area, and the fiber center coordinates values from the mesh elements
        self.area_values = np.array([elem.A for elem in self.mesh.elements])
        self.cy_values   = np.array([elem.Cy - self.mesh.Cy for elem in self.mesh.elements])
        self.cz_values   = np.array([elem.Cz - self.mesh.Cz for elem in self.mesh.elements])

        # Create groups for elements with the same material
        self.material_groups = defaultdict(list)
        self.materials = []
        for i, elem in enumerate(self.mesh.elements):
            if elem.material.name not in self.material_groups:
                self.materials.append(elem.material)

            self.material_groups[elem.material.name].append(i)

        # Initialize stress and strain array
        self.strains  = np.zeros(len(self.mesh.elements))
        self.stresses = np.zeros(len(self.mesh.elements))

    def set_strain_and_curvature(self, eps_x, xsi_y, xsi_z):
        # Set the strain and curvature values for the analysis
        self.eps_x = eps_x
        self.xsi_y = xsi_y
        self.xsi_z = xsi_z

    def calculate_strains(self):
        # Calculate strains in the pre-compiled method
        self.strains = calculate_strains_fast(self.eps_x, self.cy_values, self.cz_values, self.xsi_y, self.xsi_z)

    def calculate_stresses(self):
        # Compute stresses for each material group
        for i, (material_name, indices) in enumerate(self.material_groups.items()):
            indices = np.array(indices)  # Faster indexing
            grouped_strains = self.strains[indices]
            
            # Compute stresses efficiently using the vectorized get_stress
            self.stresses[indices] = self.materials[i].get_stress_vectorized(grouped_strains)

    def get_section_forces(self, eps_x, xsi_y, xsi_z):
        # Calculate section forces based on the provided strains and curvatures

        # Set the strain and curvature values
        self.set_strain_and_curvature(eps_x, xsi_y, xsi_z)
        # Calculate strains and stresses
        self.calculate_strains()
        self.calculate_stresses()

        # Calculate section forces using the pre-compiled method
        N, My, Mz = calculate_section_forces_fast(self.area_values, self.stresses, self.cy_values, self.cz_values)

        return N, My, Mz
    
    def get_strain_and_curvatures(self, N, My, Mz):
        # Find the strains and curvatures that result in for given section forces
        eps_x, chi_y, chi_z = fsolve(self.system_of_equations, [0.0, 0.0, 0.0], args=(N, My, Mz))
        return eps_x, chi_y, chi_z
    
    def system_of_equations(self, V, N_target, My_target, Mz_target):
        # calculate the section forces for the given strains and curvatures
        Nx, My, Mz = self.get_section_forces(V[0], V[1], V[2])
        # System of equations to solve for strains and curvatures
        return [Nx - N_target, My - My_target, Mz - Mz_target]