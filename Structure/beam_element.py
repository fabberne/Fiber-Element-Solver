from .cross_section import *
from scipy.special import legendre

class Beam_Element:
    def __init__(self, geometry, number_of_cross_sections, nodes, beam_DOFs):

        # assign the number of cross sections and initial positions of the
        self.number_of_cross_sections = number_of_cross_sections
        self.nodes_initial   = nodes
        self.nodes_displaced = nodes
        
        # get the Gauss points and weights for the beam element
        self.gauss_points, self. gauss_weights = self.get_gauss(number_of_cross_sections)

        # calculate the length of the beam element
        self.length = np.linalg.norm(self.nodes_initial[1] - self.nodes_initial[0])

        # initialize the cross sections
        self.cross_sections = []
        for i in range(number_of_cross_sections):
            self.cross_sections.append(Cross_Section(geometry, self.gauss_points[i], self.gauss_weights[i], self.length))

        # Initialize the local stiffness matrix
        self.K_local = np.zeros((5, 5))

        # Initialize the global stiffness matrix
        self.K_global = np.zeros((12, 12))

        # Initialize the resisting forces and force increment
        self.force_increment            = np.zeros((5, 1))
        self.resisting_forces           = np.zeros((5, 1))
        self.resisting_forces_converged = np.zeros((5, 1))
        self.displacements_residual     = np.zeros((5, 1))

        # set the beam DOFs
        self.beam_DOFs = np.array(beam_DOFs)


    #--------------------------------------------------------------------------------------------------------------------------------#


    def get_gauss(self, number_of_cross_sections):
        if number_of_cross_sections < 2 or number_of_cross_sections > 37:
            raise ValueError(f"number of cross sections can only be between 2 and 37")

        # Calculate the Gauss-Lobatto points as the zero points of the derivative of the Legendre polynomial
        points = list(np.sort(legendre(number_of_cross_sections - 1).deriv().roots))

        # Add the endpoints of the interval [-1, 1]
        points.insert(0, -1)
        points.append(1)
        points = np.array(points)

        # Calculate the weights using the derivative of the Legendre polynomial and following formula
        weights = 2 / (number_of_cross_sections * (number_of_cross_sections - 1) * (legendre(number_of_cross_sections - 1)(points)) ** 2)
    
        return points, weights


    def get_local_stiffness_matrix(self):
        # Initialize the local flexibility matrix
        local_flexibility_matrix = np.zeros((5, 5))

        # Calculate the jacobian
        J = self.length / 2

        # Calculate the local flexibility matrix by summing the contributions from each cross section
        for cross_section in self.cross_sections:
            local_flexibility_matrix += J * cross_section.gauss_weight * cross_section.get_global_flexibility_matrix()
        
        # Invert the local flexibility matrix to get the local stiffness matrix
        self.K_local = np.linalg.inv(local_flexibility_matrix)

        return self.K_local


    def get_global_stiffness_matrix(self):
        # get the transformation matrix and rotation matrix
        L = self.get_transformation_matrix()
        Rot = self.get_rotation_matrix()

        # get the local stiffness matrix and expand it to the global coordinates
        self.K_local  = self.get_local_stiffness_matrix()
        self.K_global = L @ self.K_local @ L.T

        # restrain the torsional DOFs
        self.K_global[3, 3] = 1
        self.K_global[9, 9] = 1

        # rotate the stiffness matrix to the global coordinate system
        self.K_global = Rot.T @ self.K_global @ Rot

        return self.K_global

    
    def get_global_resisting_forces(self):
        # takes the local resisting forces and transforms them to the global coordinate system
        L   = self.get_transformation_matrix()
        Rot = self.get_rotation_matrix()

        resisting_forces_global = L @ self.resisting_forces
        resisting_forces_global = Rot.T @ resisting_forces_global

        return resisting_forces_global


    def get_transformation_matrix(self):
        # Initialize the transformation matrix
        L = np.zeros((12, 5))

        # --- node 1 ---
        # u1: axial force N
        L[0, 4] = -1.0

        # v1: bending about z-axis
        L[1, 0] = 1.0 / self.length
        L[1, 1] = 1.0 / self.length

        # w1: bending about y-axis
        L[2, 2] = -1.0 / self.length
        L[2, 3] = -1.0 / self.length

        # θx1: torsion is not handled here

        # θy1: bending about y-axis
        L[4, 2] = 1.0

        # θz1: bending about z-axis
        L[5, 0] = 1.0

        # --- node 2 ---
        # u2: axial force N
        L[6, 4] = 1.0

        # v2: bending about z-axis
        L[7, 0] = -1.0 / self.length
        L[7, 1] = -1.0 / self.length

        # w2: bending about y-axis
        L[8, 2] = 1.0 / self.length
        L[8, 3] = 1.0 / self.length

        # θx2: torsion is not handled here

        # θy2: bending about y-axis
        L[10, 3] = 1.0

        # θz2: bending about z-axis
        L[11, 1] = 1.0

        return L

    def get_rotation_matrix(self):

        # Local x-axis (beam axis)
        v = self.nodes_initial[1] - self.nodes_initial[0]
        L = self.length

        # Unit vector along the beam axis
        e1 = v / L 

        # Global reference vector for building local z-axis
        g = np.array([0.0, 0.0, 1.0])

        # Check if e1 is parallel to global z-axis
        if np.allclose(np.abs(np.dot(e1, g)), 1.0):
            # If e1 is parallel to global z-axis, switch to global Y-axis
            g = np.array([1.0, 0.0, 0.0])

        # Build local y-axis (perpendicular to e1 and g)
        e2 = np.cross(g, e1)
        e2 /= np.linalg.norm(e2)

        # Build local z-axis (perpendicular to e1 and e2)
        e3 = np.cross(e1, e2)

        # Build small 3x3 rotation matrix R
        R = np.vstack((e1, e2, e3))

        # Build big 12x12 block-diagonal matrix Rot
        Rot = np.zeros((12, 12))
        for i in range(4):
            Rot[i*3:(i+1)*3, i*3:(i+1)*3] = R

        return Rot

    
    #--------------------------------------------------------------------------------------------------------------------------------#
    
    # the general idea of the state determination and the formulation of the iteration process is based on the work of Taucer et al. (1991)
    # [https://www.researchgate.net/publication/242435677_A_Fiber_Beam-Column_Element_for_Seismic_Response_Analysis_of_Reinforced_Concrete_Structures] 
    # and Zidan (2019) [https://zidanmahmoud.github.io/project/beamcolumn/MT.pdf]

    def state_determination(self, displacements_increment):
        T = self.get_rotation_matrix()  
        L = self.get_transformation_matrix()

        # STEP 1: Transform the displacements increment to the local coordinate system
        displacements_increment_local = T @ displacements_increment
        change_in_displacements_increment = L.T @ displacements_increment_local

        # STEP 2: start the beam iteration and...
        for j in range(self.max_section_iterations):
            print("            Element iteration ", j)

            # ... calculate the force increment
            if j == 0:
                change_in_force_increment =  self.K_local @ change_in_displacements_increment
            else:
                change_in_force_increment = -self.K_local @ self.displacements_residual

            # STEP 3: update the beam element forces
            self.force_increment += change_in_force_increment
            self.resisting_forces = self.resisting_forces_converged + self.force_increment
            
            # STEP 4 - 9: do the cross-sectional state determination 
            does_element_converge = True
            for cross_section in self.cross_sections:
                if cross_section.state_determination(change_in_force_increment):
                    pass
                else:
                    does_element_converge = False

            # STEP 10: calculate the local beam stiffness matrix
            self.K_local  = self.get_local_stiffness_matrix()

            # STEP 11: calculate the residual element deformations
            if does_element_converge:
                return
            else:
                J = self.length / 2
                self.displacements_residual.fill(0.0)
                for cross_section in self.cross_sections:
                    self.displacements_residual += J * cross_section.gauss_weight * cross_section.get_global_residuals()

        



