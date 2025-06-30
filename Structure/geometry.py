import gmsh
import numpy as np
import matplotlib.pyplot as plt

from .fiber import Fiber

# For the discription of the geometry classe and how to use the gmsh library
# see the report section 3.2 "Meshing with gmsh" or the documentation of gmsh:
# https://gmsh.info/doc/texinfo/gmsh.html#Geometry-API


class Rectangle:

    def __init__(self, width, height):
        self.width     = width
        self.height    = height


    def generate_mesh(self, type = "triangle", size = 0.1):
        gmsh.initialize()
        gmsh.model.add("rectangle")

        # Define points
        gmsh.model.geo.addPoint(-self.width/2, -self.height/2, 0, size, 1)
        gmsh.model.geo.addPoint( self.width/2, -self.height/2, 0, size, 2)
        gmsh.model.geo.addPoint( self.width/2,  self.height/2, 0, size, 3)
        gmsh.model.geo.addPoint(-self.width/2,  self.height/2, 0, size, 4)

        # Define lines
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)

        # Define surface
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)

        # Synchronize the model
        gmsh.model.geo.synchronize()

        if type == "quadrilateral":
            # Define transfinite lines (structured mesh control)
            num_div_x = int(self.width  / size)  # Number of divisions along X
            num_div_y = int(self.height / size)  # Number of divisions along Y
            gmsh.model.mesh.setTransfiniteCurve(1, num_div_x + 1)
            gmsh.model.mesh.setTransfiniteCurve(2, num_div_y + 1)
            gmsh.model.mesh.setTransfiniteCurve(3, num_div_x + 1)
            gmsh.model.mesh.setTransfiniteCurve(4, num_div_y + 1)

            # Set transfinite surface
            gmsh.model.mesh.setTransfiniteSurface(1)

            # Recombine elements to generate quadrilateral mesh
            gmsh.model.mesh.setRecombine(2,1)

        # Generate the 2D mesh
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        elements = []
        for i, e_type in enumerate(elem_types):
            if e_type == 2:
                elements = np.array(elem_nodes[i]).reshape(-1, 3) - 1
            if e_type == 3:  # Type 3 = Quadrilateral
                elements = np.array(elem_nodes[i]).reshape(-1, 4) - 1

        gmsh.finalize()
        
        elements = [Fiber(node_coords[elem], nodes=elem, mat="Concrete_C30_37") for elem in elements]

        return elements, node_coords
    
    def get_A_analytical(self):
        return self.width * self.height
    
    def get_I_x_analytical(self):
        return self.width * self.height**3 / 12
    
    def get_I_y_analytical(self):
        return self.height * self.width**3 / 12


class T_profile:

    def __init__(self, web_width, web_height, flange_width, flange_height):
        self.web_width    = web_width
        self.web_height   = web_height
        self.flange_width = flange_width
        self.flange_height= flange_height

    def generate_mesh(self, type = "triangle", size = 0.1):
        gmsh.initialize()
        gmsh.model.add("T_profile")

        # Define points
        gmsh.model.geo.addPoint(-self.web_width    / 2, -self.web_height/2 - self.flange_height/2, 0, size,  1)
        gmsh.model.geo.addPoint( self.web_width    / 2, -self.web_height/2 - self.flange_height/2, 0, size,  2)
        gmsh.model.geo.addPoint( self.web_width    / 2,  self.web_height/2 - self.flange_height/2, 0, size,  3)
        gmsh.model.geo.addPoint(-self.web_width    / 2,  self.web_height/2 - self.flange_height/2, 0, size,  4)

        gmsh.model.geo.addPoint( self.flange_width / 2,  self.web_height/2 - self.flange_height/2, 0, size,  5)
        gmsh.model.geo.addPoint( self.flange_width / 2,  self.web_height/2 + self.flange_height/2, 0, size,  6)

        
        gmsh.model.geo.addPoint( self.web_width    / 2,  self.web_height/2 + self.flange_height/2, 0, size,  7)
        gmsh.model.geo.addPoint(-self.web_width    / 2,  self.web_height/2 + self.flange_height/2, 0, size,  8)


        gmsh.model.geo.addPoint(-self.flange_width / 2,  self.web_height/2 + self.flange_height/2, 0, size,  9)
        gmsh.model.geo.addPoint(-self.flange_width / 2,  self.web_height/2 - self.flange_height/2, 0, size, 10)
        
        gmsh.model.geo.synchronize()

        # Define lines
        gmsh.model.geo.addLine(1, 2)
        gmsh.model.geo.addLine(2, 3)
        gmsh.model.geo.addLine(3, 4)
        gmsh.model.geo.addLine(4, 1)

        gmsh.model.geo.addLine(10,  4)
        gmsh.model.geo.addLine( 4,  8)
        gmsh.model.geo.addLine( 8,  9)
        gmsh.model.geo.addLine( 9, 10)

        gmsh.model.geo.addLine(3, 5)
        gmsh.model.geo.addLine(5, 6)
        gmsh.model.geo.addLine(6, 7)
        gmsh.model.geo.addLine(7, 3)

        gmsh.model.geo.addLine(7, 8)

        # Define curve loops
        web_loop           = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
        flange_loop_left   = gmsh.model.geo.addCurveLoop([5, 6, 7, 8])
        flange_loop_middle = gmsh.model.geo.addCurveLoop([-3, -12, 13, -6])
        flange_loop_right  = gmsh.model.geo.addCurveLoop([9, 10, 11, 12])

        web_surface           = gmsh.model.geo.addPlaneSurface([web_loop])
        flange_surface_left   = gmsh.model.geo.addPlaneSurface([flange_loop_left])
        flange_surface_middle = gmsh.model.geo.addPlaneSurface([flange_loop_middle])
        flange_surface_right  = gmsh.model.geo.addPlaneSurface([flange_loop_right])
        
        # Synchronize geometry
        gmsh.model.geo.synchronize()

        if type == "quadrilateral":
            # Define transfinite lines for structured meshing
            gmsh.model.mesh.setTransfiniteCurve(1, int(self.web_width  / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(2, int(self.web_height / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(3, int(self.web_width  / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(4, int(self.web_height / size) + 1)
            
            gmsh.model.mesh.setTransfiniteCurve(5, int((self.flange_width - self.web_width) / 2 / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(6, int(self.flange_height / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(7, int((self.flange_width - self.web_width) / 2 / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(8, int(self.flange_height / size) + 1)
            
            gmsh.model.mesh.setTransfiniteCurve(9, int((self.flange_width - self.web_width) / 2 / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(10, int(self.flange_height / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(11, int((self.flange_width - self.web_width) / 2 / size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(12, int(self.flange_height / size) + 1)

            gmsh.model.mesh.setTransfiniteCurve(13, int(self.web_width / size) + 1)
            
            # Set transfinite surface
            gmsh.model.mesh.setTransfiniteSurface(web_surface)
            gmsh.model.mesh.setTransfiniteSurface(flange_surface_left)
            gmsh.model.mesh.setTransfiniteSurface(flange_surface_middle)
            gmsh.model.mesh.setTransfiniteSurface(flange_surface_right)

            # Recombine into quads
            gmsh.model.mesh.setRecombine(2, web_surface)
            gmsh.model.mesh.setRecombine(2, flange_surface_left)
            gmsh.model.mesh.setRecombine(2, flange_surface_middle)
            gmsh.model.mesh.setRecombine(2, flange_surface_right)

        # Generate mesh
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        elements = []
        for i, e_type in enumerate(elem_types):
            if e_type == 2:  # Triangular elements
                elements = np.array(elem_nodes[i]).reshape(-1, 3) - 1
            if e_type == 3:  # Quadrilateral elements
                elements = np.array(elem_nodes[i]).reshape(-1, 4) - 1

        gmsh.finalize()
        
        elements = [Fiber(node_coords[elem], nodes=elem, mat="Concrete_C30_37") for elem in elements]

        return elements, node_coords


class H_beam:

    def __init__(self, web_width, web_height, flange_width, flange_height):
        self.web_width    = web_width
        self.web_height   = web_height
        self.flange_width = flange_width
        self.flange_height= flange_height

    def generate_mesh(self, type = "triangle", size = 0.1):
        gmsh.initialize()
        gmsh.model.add("H_Beam")

        # Define Points
        gmsh.model.geo.addPoint(-self.flange_width / 2,  self.web_height / 2 + self.flange_height , 0, size,  1)
        gmsh.model.geo.addPoint(-self.web_width    / 2,  self.web_height / 2 + self.flange_height , 0, size,  2)
        gmsh.model.geo.addPoint( self.web_width    / 2,  self.web_height / 2 + self.flange_height , 0, size,  3)
        gmsh.model.geo.addPoint( self.flange_width / 2,  self.web_height / 2 + self.flange_height , 0, size,  4)
        gmsh.model.geo.addPoint( self.flange_width / 2,  self.web_height / 2                      , 0, size,  5)
        gmsh.model.geo.addPoint( self.web_width    / 2,  self.web_height / 2                      , 0, size,  6)
        gmsh.model.geo.addPoint( self.web_width    / 2, -self.web_height / 2                      , 0, size,  7)
        gmsh.model.geo.addPoint( self.flange_width / 2, -self.web_height / 2                      , 0, size,  8)
        gmsh.model.geo.addPoint( self.flange_width / 2, -self.web_height / 2 - self.flange_height , 0, size,  9)
        gmsh.model.geo.addPoint( self.web_width    / 2, -self.web_height / 2 - self.flange_height , 0, size, 10)
        gmsh.model.geo.addPoint(-self.web_width    / 2, -self.web_height / 2 - self.flange_height , 0, size, 11)
        gmsh.model.geo.addPoint(-self.flange_width / 2, -self.web_height / 2 - self.flange_height , 0, size, 12)
        gmsh.model.geo.addPoint(-self.flange_width / 2, -self.web_height / 2                      , 0, size, 13)
        gmsh.model.geo.addPoint(-self.web_width    / 2, -self.web_height / 2                      , 0, size, 14)
        gmsh.model.geo.addPoint(-self.web_width    / 2,  self.web_height / 2                      , 0, size, 15)
        gmsh.model.geo.addPoint(-self.flange_width / 2,  self.web_height / 2                      , 0, size, 16)

        gmsh.model.geo.synchronize()

        # Define Lines
        gmsh.model.geo.addLine(1, 2)
        gmsh.model.geo.addLine(2, 3)
        gmsh.model.geo.addLine(3, 4)
        gmsh.model.geo.addLine(4, 5) # Top flange right 4
        gmsh.model.geo.addLine(5, 6) 
        gmsh.model.geo.addLine(6, 7) # Web right 6
        gmsh.model.geo.addLine(7, 8)
        gmsh.model.geo.addLine(8, 9) # Bottom flange right 8
        gmsh.model.geo.addLine( 9, 10)
        gmsh.model.geo.addLine(10, 11)
        gmsh.model.geo.addLine(11, 12)
        gmsh.model.geo.addLine(12, 13) # Bottom flange left 12
        gmsh.model.geo.addLine(13, 14)
        gmsh.model.geo.addLine(14, 15) # Web left 14 
        gmsh.model.geo.addLine(15, 16)
        gmsh.model.geo.addLine(16,  1) # Top flange left 16

        gmsh.model.geo.addLine( 6, 15)  # Vertical web top
        gmsh.model.geo.addLine( 7, 14)  # Vertical web bottom

        gmsh.model.geo.addLine( 2, 15)  # top flange left 19
        gmsh.model.geo.addLine( 3, 6)   # top flange right 20

        gmsh.model.geo.addLine(11, 14)  # bottom flange left
        gmsh.model.geo.addLine( 7, 10)   # bottom flange right

        # Add curve loops
        top_flange_left   = gmsh.model.geo.addCurveLoop([1, 19, 15, 16])
        top_flange_middle = gmsh.model.geo.addCurveLoop([2, 20, 17,-19])
        top_flange_right  = gmsh.model.geo.addCurveLoop([3,  4,  5,-20])

        bottom_flange_left   = gmsh.model.geo.addCurveLoop([ 13,-21, 11, 12])
        bottom_flange_middle = gmsh.model.geo.addCurveLoop([-18, 22, 10, 21])
        bottom_flange_right  = gmsh.model.geo.addCurveLoop([  7,  8,  9,-22])

        web_loop = gmsh.model.geo.addCurveLoop([-17, 6, 18, 14])
        
        # Define Plane Surfaces
        top_flange_left_surface   = gmsh.model.geo.addPlaneSurface([top_flange_left])
        top_flange_middle_surface = gmsh.model.geo.addPlaneSurface([top_flange_middle])
        top_flange_right_surface  = gmsh.model.geo.addPlaneSurface([top_flange_right])

        bottom_flange_left_surface   = gmsh.model.geo.addPlaneSurface([bottom_flange_left])
        bottom_flange_middle_surface = gmsh.model.geo.addPlaneSurface([bottom_flange_middle])
        bottom_flange_right_surface  = gmsh.model.geo.addPlaneSurface([bottom_flange_right])

        web_surface = gmsh.model.geo.addPlaneSurface([web_loop])

        # Synchronize the model
        gmsh.model.geo.synchronize()

        # Apply structured meshing for quadrialteral elements
        if type == "quadrilateral":
            for curve in [4,8,12,16,19,20,21,22]:
                gmsh.model.mesh.setTransfiniteCurve(curve, int(self.flange_height / size) + 1)
            for curve in [1,3,5,7,9,11,13,15]:
                gmsh.model.mesh.setTransfiniteCurve(curve, int((self.flange_width - self.web_width) / 2 / size) + 1)
            for curve in [2,10,17,18]:
                gmsh.model.mesh.setTransfiniteCurve(curve, int(self.web_width  / size) + 1)
            for curve in [6,14]:
                gmsh.model.mesh.setTransfiniteCurve(curve, int(self.web_height / size) + 1)

            gmsh.model.mesh.setTransfiniteSurface(top_flange_left_surface)
            gmsh.model.mesh.setTransfiniteSurface(top_flange_middle_surface)
            gmsh.model.mesh.setTransfiniteSurface(top_flange_right_surface)
            gmsh.model.mesh.setTransfiniteSurface(bottom_flange_left_surface)
            gmsh.model.mesh.setTransfiniteSurface(bottom_flange_middle_surface)
            gmsh.model.mesh.setTransfiniteSurface(bottom_flange_right_surface)
            gmsh.model.mesh.setTransfiniteSurface(web_surface)

            gmsh.model.mesh.setRecombine(2, top_flange_left_surface)
            gmsh.model.mesh.setRecombine(2, top_flange_middle_surface)
            gmsh.model.mesh.setRecombine(2, top_flange_right_surface)
            gmsh.model.mesh.setRecombine(2, bottom_flange_left_surface)
            gmsh.model.mesh.setRecombine(2, bottom_flange_middle_surface)
            gmsh.model.mesh.setRecombine(2, bottom_flange_right_surface)
            gmsh.model.mesh.setRecombine(2, web_surface)

        # Generate mesh
        gmsh.model.mesh.generate(2)

        # Extract nodes and elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        elements = []
        for i, e_type in enumerate(elem_types):
            if e_type == 2:  # Triangular elements
                elements = np.array(elem_nodes[i]).reshape(-1, 3) - 1
            if e_type == 3:  # Quadrilateral elements
                elements = np.array(elem_nodes[i]).reshape(-1, 4) - 1

        gmsh.finalize()
        
        elements = [Fiber(node_coords[elem], nodes=elem, mat="Steel_S235") for elem in elements]

        return elements, node_coords


class Polygon:

    def __init__(self, points):
        self.points = points

    def generate_mesh(self, type = "triangle", size = 0.1):
        gmsh.initialize()
        gmsh.model.add("Polygon")

        # Define Points
        for i, (x, y) in enumerate(self.points):
            gmsh.model.geo.addPoint(x, y, 0, size, i + 1)

        gmsh.model.geo.synchronize()

        # Define Lines
        for i in range(len(self.points) - 1):
            gmsh.model.geo.addLine(i + 1, i + 2)
        
        gmsh.model.geo.addLine(len(self.points), 1)

        # Define Curve Loop
        gmsh.model.geo.addCurveLoop(list(range(1, len(self.points) + 1)))
        
        # Define Plane Surfaces
        gmsh.model.geo.addPlaneSurface([1])

        gmsh.model.geo.synchronize()

        if type == "quadrilateral":
            # Ensure quadrilateral elements
            gmsh.model.mesh.setRecombine(2, 1)

        # Generate and optimize mesh
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        elements = []
        for i, e_type in enumerate(elem_types):
            if e_type == 2:
                elements = np.array(elem_nodes[i]).reshape(-1, 3) - 1
            if e_type == 3:
                elements = np.array(elem_nodes[i]).reshape(-1, 4) - 1

        gmsh.finalize()

        elements = [Fiber(node_coords[elem], nodes=elem) for elem in elements]

        return elements, node_coords

class L_beam:

    def __init__(self, width, height, thickness, r_corner, r_edge, rotation_angle = 0):
        self.h      = height
        self.w      = width
        self.r_1    = r_corner
        self.r_2    = r_edge
        self.t      = thickness
        self.cos    = np.cos(np.radians(rotation_angle))
        self.sin    = np.sin(np.radians(rotation_angle))

        self.points = [(0                , 0                ),
                       (self.w           , 0                ),
                       (self.w           , self.t - self.r_2),
                       (self.w - self.r_2, self.t - self.r_2),
                       (self.w - self.r_2, self.t           ),
                       (self.t + self.r_1, self.t           ),
                       (self.t + self.r_1, self.t + self.r_1),
                       (self.t           , self.t + self.r_1),
                       (self.t           , self.h - self.r_2),
                       (self.t - self.r_2, self.h - self.r_2),
                       (self.t - self.r_2, self.h           ),
                       (0                , self.h           )]

        for i, (x, y) in enumerate(self.points):
            self.points[i] = (x * self.cos - y * self.sin,
                              x * self.sin + y * self.cos)

    def generate_mesh(self, type = "triangle", size = 0.1):
        gmsh.initialize()
        gmsh.model.add("Polygon")

        # Define Points
        for i, (x, y) in enumerate(self.points):
            gmsh.model.geo.addPoint(x, y, 0, size, i + 1)

        gmsh.model.geo.synchronize()

        # Define Lines
        gmsh.model.geo.addLine(1, 2)
        gmsh.model.geo.addLine(2, 3)

        gmsh.model.geo.addCircleArc(3, 4, 5)

        gmsh.model.geo.addLine(5, 6)

        gmsh.model.geo.addCircleArc(6, 7, 8)

        gmsh.model.geo.addLine(8, 9)

        gmsh.model.geo.addCircleArc(9, 10, 11)

        gmsh.model.geo.addLine(11, 12)
        gmsh.model.geo.addLine(12, 1)

        # Define Curve Loop
        gmsh.model.geo.addCurveLoop(list(range(1, 10)))
        
        # Define Plane Surfaces
        gmsh.model.geo.addPlaneSurface([1])

        gmsh.model.geo.synchronize()

        if type == "quadrilateral":
            # Ensure quadrilateral elements
            gmsh.model.mesh.setRecombine(2, 1)

        # Generate and optimize mesh
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        elements = []
        for i, e_type in enumerate(elem_types):
            if e_type == 2:
                elements = np.array(elem_nodes[i]).reshape(-1, 3) - 1
            if e_type == 3:
                elements = np.array(elem_nodes[i]).reshape(-1, 4) - 1

        gmsh.finalize()

        elements = [Fiber(node_coords[elem], nodes=elem, mat="Steel_S235") for elem in elements]

        return elements, node_coords

class ReinforcedConcreteColumn:
    def __init__(self, width, height, concrete_cover, rebar_diameter, rebar_spacing):
        self.width  = width
        self.height = height

        self.concrete_cover = concrete_cover
        self.rebar_diameter = rebar_diameter
        self.rebar_spacing  = rebar_spacing

    def generate_mesh(self, type="triangle", size=0.1):

        if type != "triangle":
            raise ValueError("Only triangular elements are supported for this geometry")
        
        gmsh.initialize()
        gmsh.model.add("rectangle_with_hole")

        # Define concrete points
        gmsh.model.geo.addPoint(-self.width/2,-self.height/2, 0, size, 1)
        gmsh.model.geo.addPoint( self.width/2,-self.height/2, 0, size, 2)
        gmsh.model.geo.addPoint( self.width/2, self.height/2, 0, size, 3)
        gmsh.model.geo.addPoint(-self.width/2, self.height/2, 0, size, 4)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)

        outer_loop = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
        
        # Define rebar
        number_of_rebars = int((self.width - 2 * self.concrete_cover - self.rebar_diameter) / self.rebar_spacing)
        rebar_positions_x = np.linspace(-self.width/2 + self.concrete_cover + self.rebar_diameter/2, 
                                         self.width/2 - self.concrete_cover - self.rebar_diameter/2, 
                                         number_of_rebars)
        rebar_positions_x = np.concatenate([rebar_positions_x, rebar_positions_x])

        rebar_positions_y_1 = np.ones(number_of_rebars) * self.height/2 - self.concrete_cover - self.rebar_diameter/2
        rebar_positions_y_2 = np.ones(number_of_rebars) *-self.height/2 + self.concrete_cover + self.rebar_diameter/2
        rebar_positions_y   = np.concatenate([rebar_positions_y_1, rebar_positions_y_2])

        d = self.rebar_diameter
        for i, (x, y) in enumerate(zip(rebar_positions_x, rebar_positions_y)):
            gmsh.model.geo.addPoint(x, y, 0, d, (i+1) * 10)

            gmsh.model.geo.addPoint(x + d/2, y, 0, d, (i+1) * 10 + 1)
            gmsh.model.geo.addPoint(x - d/2, y, 0, d, (i+1) * 10 + 2)
            gmsh.model.geo.addPoint(x, y + d/2, 0, d, (i+1) * 10 + 3)
            gmsh.model.geo.addPoint(x, y - d/2, 0, d, (i+1) * 10 + 4)
        
            gmsh.model.geo.addCircleArc((i+1) * 10 + 1, (i+1) * 10, (i+1) * 10 + 3, (i+1) * 100 + 0)
            gmsh.model.geo.addCircleArc((i+1) * 10 + 3, (i+1) * 10, (i+1) * 10 + 2, (i+1) * 100 + 1)
            gmsh.model.geo.addCircleArc((i+1) * 10 + 2, (i+1) * 10, (i+1) * 10 + 4, (i+1) * 100 + 2)
            gmsh.model.geo.addCircleArc((i+1) * 10 + 4, (i+1) * 10, (i+1) * 10 + 1, (i+1) * 100 + 3)

        rebar_loops    = [gmsh.model.geo.addCurveLoop([(i+1) * 100 + 0, (i+1) * 100 + 1, (i+1) * 100 + 2, (i+1) * 100 + 3]) for i in range(number_of_rebars * 2)]

        # Create plane surface
        concrete_surface = gmsh.model.geo.addPlaneSurface([outer_loop] + rebar_loops)
        rebar_surfaces   = [gmsh.model.geo.addPlaneSurface([loop]) for loop in rebar_loops]
        gmsh.model.geo.synchronize()

        # Assign physical groups
        concrete_group = gmsh.model.addPhysicalGroup(2, [concrete_surface])
        rebar_group    = gmsh.model.addPhysicalGroup(2, rebar_surfaces)

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements()

        concrete_elements = set(gmsh.model.mesh.getElementsByType(2, concrete_surface)[0])
        rebar_elements    = []
        for surface in rebar_surfaces:
            rebar_elements.append(list(gmsh.model.mesh.getElementsByType(2, surface)[0]))
        rebar_elements = set([item for sublist in rebar_elements for item in sublist])

        gmsh.finalize()

        element_nodes = []
        for i, e_type in enumerate(elem_types):
            if e_type == 2:
                element_nodes = np.array(elem_nodes[i]).reshape(-1, 3) - 1
            if e_type == 3:
                element_nodes = np.array(elem_nodes[i]).reshape(-1, 4) - 1

        elements = []
        for i, elem in enumerate(element_nodes):
            if type == "triangle":
                nodes = np.where((node_tags == elem[0] + 1) | (node_tags == elem[1] + 1) | (node_tags == elem[2] + 1))[0]
            else:
                nodes = np.where((node_tags == elem[0] + 1) | (node_tags == elem[1] + 1) | (node_tags == elem[2] + 1) | (node_tags == elem[3] + 1))[0]

            if i+1 in concrete_elements:
                elements.append(Fiber(node_coords[nodes], nodes=nodes, mat="Concrete_C30_37"))
            elif i+1 in rebar_elements:
                elements.append(Fiber(node_coords[nodes], nodes=nodes, mat="Rebar_B500B"))
            else:
                elements.append(Fiber(node_coords[nodes], nodes=nodes, mat="unknown"))

        return elements, node_coords