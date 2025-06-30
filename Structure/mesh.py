import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
from tabulate import tabulate

class Mesh:

    def __init__(self, geometry, mesh_type, mesh_size):
        # store the mesh type and and approximate mesh size
        # mesh_type can be "triangular" or "quadrilateral"
        # mesh_size is the approximate distance between the nodes in mm
        self.mesh_type = mesh_type
        self.mesh_size = mesh_size

        # generate the mesh using the geometry object
        elements, node_coords = geometry.generate_mesh(mesh_type, mesh_size)

        # store the elements and the nodal coordinates
        self.elements    = elements
        self.node_coords = node_coords

        # compute the area, centroid and inertia of the mesh respectively the cross section
        self.A           = self.get_A_numerical()
        self.Cy, self.Cz = self.get_centroid()
        self.Iy, self.Iz = self.get_I_numerical()

        
    #--------------------------------------------------------------------------------------------------------------------------------#


    def get_A_numerical(self):
        # Calculate the total area of the mesh by summing the areas of all fibers
        total_area = sum(elem.A for elem in self.elements)
        return total_area


    def get_centroid(self):
        # Calculate the centroid coordinates by summing the weighted coordinates of all fibers
        c_y = sum(elem.A * elem.Cy for elem in self.elements) / self.A
        c_z = sum(elem.A * elem.Cz for elem in self.elements) / self.A
        return c_y, c_z 
    
    
    def get_I_numerical(self):
        Iy = 0
        Iz = 0

        # Calculate the moments of inertia using the parallel axis theorem
        for elem in self.elements:
            Iy += elem.A * (elem.Cz - self.Cz)**2 + elem.Iy
            Iz += elem.A * (elem.Cy - self.Cy)**2 + elem.Iz

        return Iy, Iz

        
    #--------------------------------------------------------------------------------------------------------------------------------#


    def print(self):
        # Print the mesh properties in a formatted table
        Mesh_properties = [("Mesh Type"         , self.mesh_type       ),
                           ("Number of elements", len(self.elements   )),
                           ("Number of nodes"   , len(self.node_coords)),
                           ("Cross Section Area", str.format('{0:.2f}', self.A))]

        print(tabulate(Mesh_properties, 
                       tablefmt = "fancy_grid",disable_numparse=True))

        CS_properties = [(" ", "y", "z")]
    
        CS_properties.append(("Centroid [mm]"           , format_value(self.Cy), format_value(self.Cz)))
        CS_properties.append(("Moment of inertia [mm^4]", format_value(self.Iy), format_value(self.Iz)))
    
        print(tabulate(CS_properties, 
                       headers  =   "firstrow", 
                       tablefmt = "fancy_grid",disable_numparse=True))


    def plot(self):
        # Plot the mesh with the elements and nodes
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        
        fig, ax = plt.subplots(figsize=(4, 4))

        for i, element in enumerate(self.elements):
            y = element.coords[:, 0]
            z = element.coords[:, 1]
            poly = patches.Polygon(np.column_stack([y, z]), 
                                    edgecolor = element.material.color,
                                    facecolor = element.material.color, 
                                    lw        = 0.3)
            ax.add_patch(poly)


        ax.scatter(self.node_coords[:, 0], 
                   self.node_coords[:, 1], 
                   c     =   'red', 
                   s     =       2, 
                   label = "Nodes")
        ax.scatter(self.Cy, 
                   self.Cz, 
                   c     =    "blue", 
                   s     =        20, 
                   label = "Centroid")

        ax.set_xlabel("y [$mm$]")
        ax.set_ylabel("z [$mm$]")
        ax.set_frame_on(False)
        ax.set_aspect('equal')
        ax.legend()
        plt.show()


def format_value(val):
    # Format the value for better readability
    if isinstance(val, float):
        if abs(val) < 1e-10:
            return "0.00"
        elif abs(val) > 10000:
            return f"{val:.2e}"
        else:
            return f"{val:.2f}"
    return val