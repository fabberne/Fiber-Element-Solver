import numpy as np
from collections import defaultdict

from .fiber import Fiber

class Cross_Section:
    def __init__(self, geometry, gauss_point, gauss_weight, beam_element_length):
        # initizalize the geometry which is of type Mesh
        self.geometry = geometry

        # setup the array of all fibers
        self.fibers = geometry.elements

        # store the element length of the beam
        self.beam_element_length = beam_element_length

        # set the cross-sections gauss-weight and gauss-point
        self.gauss_point  = gauss_point
        self.gauss_weight = gauss_weight

        # Create groups for elements with the same material
        self.material_groups = defaultdict(list)
        self.materials = []
        for i, fiber in enumerate(self.fibers):
            if fiber.material.name not in self.material_groups:
                self.materials.append(fiber.material)
            self.material_groups[fiber.material.name].append(i)

        # setup fiber properties
        self.fibers_y_coord = np.zeros(len(self.fibers))
        self.fibers_z_coord = np.zeros(len(self.fibers))
        self.fibers_A       = np.zeros(len(self.fibers))
        
        # initialize the necessary arrays for the cross-section
        self.fibers_tangent_modulus = np.zeros(len(self.fibers))
        self.strains                = np.zeros(len(self.fibers))
        self.strains_converged      = np.zeros(len(self.fibers))
        self.strains_increment      = np.zeros(len(self.fibers))
        self.stresses               = np.zeros(len(self.fibers))

        # store the fibers coordinates and areas for vectorized operations
        for i, fiber in enumerate(self.fibers):
            self.fibers_y_coord[i] = fiber.Cy
            self.fibers_z_coord[i] = fiber.Cz
            self.fibers_A[i]       = fiber.A

        # set initial tangent modulus for each material
        for i, (material_name, indices) in enumerate(self.material_groups.items()):
            indices = np.array(indices)
            grouped_strains = self.strains[indices]
            
            # Compute stresses efficiently using the vectorized get_stress
            self.fibers_tangent_modulus[indices] = self.materials[i].get_tangent_vectorized(grouped_strains)
        
        # set initial flexibility matrix
        self.section_flex_matrix = self.get_flexibility_matrix()

        # compute the force interpolation matrix
        self.section_b_matrix = np.zeros((3, 5))
        self.section_b_matrix[0, 0] = self.gauss_point / 2 - 1 / 2
        self.section_b_matrix[0, 1] = self.gauss_point / 2 + 1 / 2
        self.section_b_matrix[1, 2] = self.gauss_point / 2 - 1 / 2
        self.section_b_matrix[1, 3] = self.gauss_point / 2 + 1 / 2
        self.section_b_matrix[2, 4] = 1

        # initialize interation variables
        self.forces_increment  = np.zeros((3, 1))
        self.forces            = np.zeros((3, 1))
        self.forces_converged  = np.zeros((3, 1))
        self.unbalanced_forces = np.zeros((3, 1))
        self.residuals         = np.zeros((3, 1))
        self.curvature         = np.zeros((3, 1))


    #--------------------------------------------------------------------------------------------------------------------------------#


    def get_flexibility_matrix(self):
        # get the tangent modulus for each material group
        for i, (material_name, indices) in enumerate(self.material_groups.items()):
            indices = np.array(indices)
            grouped_strains = self.strains[indices]
            
            # Compute stresses efficiently using the vectorized get_stress
            self.fibers_tangent_modulus[indices] = self.materials[i].get_tangent_vectorized(grouped_strains)

        # initialize the section stiffness matrix
        section_K = np.zeros((3, 3))
        
        # loop over all fibers and calculate the section stiffness matrix
        for i, fiber in enumerate(self.fibers):
            section_K[0,0] += self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_y_coord[i]**2
            section_K[1,1] += self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_z_coord[i]**2
            section_K[2,2] += self.fibers_tangent_modulus[i] * self.fibers_A[i]

            section_K[0,1] -= self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_y_coord[i] * self.fibers_z_coord[i]
            section_K[1,0] -= self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_y_coord[i] * self.fibers_z_coord[i]

            section_K[0,2] -= self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_y_coord[i]
            section_K[2,0] -= self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_y_coord[i]

            section_K[1,2] += self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_z_coord[i]
            section_K[2,1] += self.fibers_tangent_modulus[i] * self.fibers_A[i] * self.fibers_z_coord[i]

        # invert the section stiffness matrix to get the flexibility matrix
        self.section_flex_matrix = np.linalg.inv(section_K)

        return self.section_flex_matrix


    def get_global_flexibility_matrix(self):
        # calculate the flexibility matrix for the section and expand it to the beam coordinates using the force interpolation matrix
        self.section_flex_matrix = self.get_flexibility_matrix()
        self.section_global_flex_matrix = self.section_b_matrix.T @ self.section_flex_matrix @ self.section_b_matrix

        return self.section_global_flex_matrix


    def get_global_residuals(self):
        # calculate the global residuals by multiplying the force interpolation matrix with the unbalanced forces
        return self.section_b_matrix.T @ self.residuals

    
    #--------------------------------------------------------------------------------------------------------------------------------#
    
    # the general idea of the state determination and the formulation of the iteration process is based on the work of Taucer et al. (1991)
    # [https://www.researchgate.net/publication/242435677_A_Fiber_Beam-Column_Element_for_Seismic_Response_Analysis_of_Reinforced_Concrete_Structures] 
    # and Zidan (2019) [https://zidanmahmoud.github.io/project/beamcolumn/MT.pdf]

    def state_determination(self, change_in_force_increment):
        
        # STEP 4: calculate the force increment in the cross-sectional coordinates and update the section force vector
        change_in_force_increment = self.section_b_matrix @ change_in_force_increment
        self.forces_increment += change_in_force_increment
        self.forces = self.forces_converged + self.forces_increment

        # STEP 5: calculate the section deformation increment
        change_in_deformation_increment = self.residuals + self.section_flex_matrix @ change_in_force_increment

        # store the deformations for later plots and visualizations
        self.curvature += change_in_deformation_increment

        # STEP 6: calculate the section defoirmations respectively the fiber strains
        self.change_in_strain_increment = (- self.fibers_y_coord * change_in_deformation_increment[0] 
                                           + self.fibers_z_coord * change_in_deformation_increment[1]
                                           + change_in_deformation_increment[2])

        self.strains_increment += self.change_in_strain_increment
        self.strains = self.strains_converged + self.strains_increment

        # STEP 7: update the fiber stresses per material group
        for i, (material_name, indices) in enumerate(self.material_groups.items()):
            indices = np.array(indices) 
            grouped_strains = self.strains[indices]
            
            # Compute stresses efficiently using the vectorized get_stress
            self.stresses[indices] = self.materials[i].get_stress_vectorized(grouped_strains)

        # STEP 7: update the cross sections flexibility matrix
        self.section_flex_matrix = self.get_flexibility_matrix()

        # STEP 8: calculate the resisting forces
        resisting_forces = np.zeros((3, 1))
        for i, fiber in enumerate(self.fibers):
            resisting_forces[0] -= self.stresses[i] * self.fibers_A[i] * self.fibers_y_coord[i]
            resisting_forces[1] += self.stresses[i] * self.fibers_A[i] * self.fibers_z_coord[i]
            resisting_forces[2] += self.stresses[i] * self.fibers_A[i]
        
        # STEP 9: calculate the section unbalanced forces and the residual deformations
        self.unbalanced_forces = self.forces - resisting_forces
        self.residuals = self.section_flex_matrix @ self.unbalanced_forces

        # STEP 10: check if the cross-section converged
        return abs(np.linalg.norm(self.unbalanced_forces)) < self.tolerance