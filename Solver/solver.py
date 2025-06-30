import numpy as np
import pandas as pd
import tqdm

from .constraint import *

class Solver():

    def print_nodal_displacements_and_forces(self):
        # Define labels for the DOFs
        dof_labels   = [ 'u [mm]',  'v [mm]',  'w [mm]', 'θx [rad]', 'θy [rad]', 'θz [rad]']
        F_int_labels = ['Fx [kN]', 'Fy [kN]', 'Fz [kN]', 'Mx [kNm]', 'My [kNm]', 'Mz [kNm]']

        # Print the displacements and internal forces for each node
        for node in range(self.structure.number_of_nodes):
            # Get the displacements and internal forces for the current node
            # Reshape the displacements and forces to 1D arrays
            u_node     = self.displacements[node*6:(node+1)*6].reshape(6)
            F_int_node = self.forces[       node*6:(node+1)*6].reshape(6)

            # convert internal forces to kN and bending moments to kNm
            F_int_node[[0,1,2]] = F_int_node[[0,1,2]] / 1000.0    # Convert to kN
            F_int_node[[3,4,5]] = F_int_node[[3,4,5]] / 1000000.0 # Convert to kNm

            # Create DataFrames for displacements and internal forces
            df1 = pd.DataFrame({'Displacement'  : u_node    }, index =   dof_labels)
            df2 = pd.DataFrame({'Internal Force': F_int_node}, index = F_int_labels)

            # Print the DataFrames with specified formatting
            pd.set_option('display.precision', 9)
            pd.set_option('display.float_format', '{:,.6f}'.format)
            print("--------------------------------------")
            print("Node", node)
            print(df1)
            print(df2)


#--------------------------------------------------------------------------------------------------------------------------------#


class Linear(Solver):
    def __init__(self, structure):
        # Initialize the linear solver with the given structure
        self.structure = structure
 
        # Initialize arrays for displacements and forces
        self.displacements = np.zeros(self.structure.number_of_DOFs)
        self.forces        = np.zeros(self.structure.number_of_DOFs)

    def solve(self):
        # Assemble the global stiffness matrix
        self.structure.assemble()

        # Solve the system of equations to find the displacements
        self.displacements = np.linalg.solve(self.structure.K_global, self.structure.F_global)

        # reassemble the structure without boundary conditions to get the reaction forces
        self.structure.assemble_without_bc()
        self.forces = np.dot(self.structure.K_global, self.displacements)

        # print the nodal displacements and forces
        self.print_nodal_displacements_and_forces()

        return
    

#--------------------------------------------------------------------------------------------------------------------------------#


class Nonlinear(Solver):
    def __init__(self, structure, constraint = "Load", 
                 NR_tolerance      = 1e-6, NR_max_iter      = 100, 
                 section_tolerance = 1e-6, section_max_iter = 100, 
                 controlled_DOF=None):

        # Initialize the nonlinear solver with the given structure
        self.structure = structure

        # define the maximum number of attempts to reach convergence, where at each attempt the increment is halved
        self.attempts = 5

        # Set the tolerance and maximum iterations for the Newton-Raphson method
        self.NR_tolerance = NR_tolerance
        self.NR_max_iter  = NR_max_iter

        # Set the tolerance and maximum iterations for the section state determination
        self.structure.set_section_max_iter_and_tolerance(section_max_iter, section_tolerance)

        # Initialize the constraint based on the specified type
        if constraint == "Load":
            self.constraint = Load()
        elif constraint == "Displacement":
            self.constraint = Displacement()
        elif constraint == "Arc":
            self.constraint = Arc()
        else:
            raise ValueError(f"Unknown constraint type: '{constraint}'. Expected one of 'Load', 'Displacement', or 'Arc'.")
        
        # Set the index of the controlled degree of freedom (DOF) if specified for the displacement control
        self.controlled_DOF = controlled_DOF


    def solve(self, increments):
        # initialize arrays to store the results
        # lambda_history:          load factor at each load step     [load step]
        # u_history:               displacements at each load step   [load step, DOF]
        # section_forces_history:  section forces at each load step  [load step, number of beams, number of cross sections, [Mz, My, N]]
        # section_strains_history: section strains at each load step [load step, number of beams, number of cross sections, [kappa_z, kappa_y, delta]]

        lambda_history = np.zeros(len(increments) + 1)
        u_history      = np.zeros((len(increments) + 1, self.structure.number_of_DOFs))
        section_forces_history  = np.zeros((len(increments), len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))
        section_strains_history = np.zeros((len(increments), len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))

        # top level iteration loop for each load step
        for step in tqdm.tqdm_notebook(range(len(increments))):
            print("----------------------------------------------")
            print("Load step", step + 1, "of", len(increments))
            attempt = 1
            convergence_boolean = False

            # STEP 1: get the initial state of the structure
            u0, lambda0 = self.structure.getState()
            lambda0     = float(lambda0)
            u0          = u0.copy()

            while (not convergence_boolean) and (attempt <= self.attempts):
                print("   Attempt ", attempt)
                attempt += 1

                # STEP 2 - 13: compute the newton-raphson solution for the current load step
                u, llambda, convergence_boolean, section_forces, section_strains = self.getSolution(u0, lambda0, 
                                                                                                    increments[step], 
                                                                                                    self.controlled_DOF)

                if (not convergence_boolean) and (attempt <= self.attempts):
                    increments[step] *= 0.5
                    print("   Decreased increment to ", increments[step])

            if not convergence_boolean:
                print("   Failed to reach convergence after ", attempt-1, "attempts")

            # update the state of the structure with the converged solution
            u_history[step + 1, :]      = u.reshape(self.structure.number_of_DOFs)
            lambda_history[step + 1   ] = llambda
            section_forces_history[ step, :, :, :] = np.array(section_forces)
            section_strains_history[step, :, :, :] = np.array(section_strains)

        return u_history, lambda_history, section_forces_history, section_strains_history


    def getSolution(self, u0, lambda0, increment, controlled_DOF=None):
        # STEP 2: Get the structural matrices
        self.structure.assemble()
        Stiffness_K, fext, ResidualsR = self.structure.K_global, self.structure.F_global, self.structure.Residual

        # initialize the norm fot the convergence criteria
        convergence_norm = max(np.linalg.norm(fext),1)

        # STEP 3 - 6: prediction of the deformation and load factor
        (u, llambda, deltaUp, deltalambdap, Stiffness_K, fext, ResidualsR,
        ) = self.constraint.predict(self.structure.getSystemMatrices,
                                    u0, lambda0, increment, Stiffness_K, 
                                    fext, ResidualsR)

        section_forces  = np.zeros((len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))
        section_strains = np.zeros((len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))
        
        # STEP 7: start the Newton-Raphson iterations
        for iteration in range(self.NR_max_iter):

            # STEP 8: compute the constraint values depending on the type of constraint
            print("      NR Iteration ", iteration)
            g, h, s = self.constraint.get(u, llambda, u0, lambda0, 
                                          deltaUp, deltalambdap,
                                          increment, controlled_DOF=controlled_DOF)

            # STEP 9: solving the system of equations
            du_tilde        =   np.linalg.inv(Stiffness_K).dot(fext)
            du_double_tilde = - np.linalg.inv(Stiffness_K).dot(ResidualsR)

            # STEP 10: calculation of the load factor increment and deformation increment
            deltalambdap    = - (g + np.transpose(h).dot(du_double_tilde)) / (s +np.transpose(h).dot(du_tilde))
            deltaUp         = deltalambdap * du_tilde + du_double_tilde

            # STEP 11: update the structure with the new increments, here happens the cross-section state determination 
            Stiffness_K, fext, ResidualsR = self.structure.getSystemMatrices(deltaUp, deltalambdap)
        
            # STEP 12: update the solution variables
            u       = self.structure.displacements
            llambda = self.structure.lambda_factor_converged
            
            # STEP 13: check convergence
            print("      Residuals Norm ", np.linalg.norm(ResidualsR))
            if np.linalg.norm(ResidualsR) <= self.NR_tolerance * convergence_norm:
                print("NR Converged!")
                convergence_boolean = True

                # if converged: finalize the load step
                self.structure.displacements_increment.fill(0.0)
                self.structure.lambda_factor_increment = 0.0
                self.structure.displacements_converged = self.structure.displacements
                self.structure.lambda_factor_converged = self.structure.lambda_factor
                for i, beam_element in enumerate(self.structure.beam_elements):
                    beam_element.resisting_forces_converged = beam_element.resisting_forces
                    beam_element.force_increment.fill(0.0)

                    for j, section in enumerate(beam_element.cross_sections):
                        section.forces_converged = section.forces
                        section.forces_increment.fill(0.0)
                        section_forces[i, j, :]  = section.forces_converged.reshape(3)
                        section_strains[i, j, :] = section.curvature.reshape(3)

                        section.strains_converged = section.strains
                        section.strains_increment.fill(0.0)

                break
            else:
                convergence_boolean = False
                
        if self.constraint.name == "Displacement control":
            return -u, -llambda, convergence_boolean, section_forces, section_strains
        else:
            return  u,  llambda, convergence_boolean, section_forces, section_strains