import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
from scipy.optimize import fsolve

def plot_stress_strain_steel(analysis):

	# Normalize strains for color mapping
	max_strain = max(abs(min(analysis.strains*100)), abs(max(analysis.strains*100)))
	norm = colors.TwoSlopeNorm(vmin=-max_strain, vcenter=0, vmax=max_strain)
	cmap = plt.colormaps.get_cmap('coolwarm')

	fig, (strains, stresses) = plt.subplots(1,2, figsize=(8, 4))
	
	for i, elem in enumerate(analysis.mesh.elements):
		x = elem.coords[:, 0]
		y = elem.coords[:, 1]
		poly = patches.Polygon(np.column_stack([x, y]),
								edgecolor='black',
								facecolor=cmap(norm(analysis.strains[i]*100)),
								lw=0.3)
		strains.add_patch(poly)

	# Plotting nodes to improve visual correctness without marking them
	strains.plot(analysis.mesh.node_coords[:, 0],
			analysis.mesh.node_coords[:, 1],
			'o', markersize=0, color='black')

	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	plt.colorbar(sm, ax=strains, label=r"strain $\varepsilon$ [%]", shrink=0.7)

	strains.set_frame_on(False)
	strains.set_title("strain visualization")
	strains.set_xlim(analysis.mesh.node_coords[:, 0].min(), analysis.mesh.node_coords[:, 0].max())
	strains.set_ylim(analysis.mesh.node_coords[:, 1].min(), analysis.mesh.node_coords[:, 1].max())
	strains.axes.get_xaxis().set_ticks([])
	strains.axes.get_yaxis().set_ticks([])
	strains.set_aspect('equal')

	# Normalize stresses for color mapping
	max_strain = max(abs(min(analysis.stresses)), abs(max(analysis.stresses)))
	norm = colors.TwoSlopeNorm(vmin=-max_strain, vcenter=0, vmax=max_strain)
	cmap = plt.colormaps.get_cmap('coolwarm')

	for i, elem in enumerate(analysis.mesh.elements):
		x = elem.coords[:, 0]
		y = elem.coords[:, 1]
		if abs(analysis.stresses[i]) >= 235:
			poly = patches.Polygon(np.column_stack([x, y]),
									edgecolor='black',
									facecolor=cmap(norm(analysis.stresses[i])),
									lw=0.4)
		else:
			poly = patches.Polygon(np.column_stack([x, y]),
									edgecolor='black',
									facecolor=cmap(norm(analysis.stresses[i])),
									lw=0)
		stresses.add_patch(poly)

	# Plotting nodes to improve visual correctness without marking them
	stresses.plot(analysis.mesh.node_coords[:, 0],
			analysis.mesh.node_coords[:, 1],
			'o', markersize=0, color='black')

	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	plt.colorbar(sm, ax=stresses, label=r"stress $\sigma$ [MPa]", shrink=0.7)

	stresses.set_frame_on(False)
	stresses.set_title("stress visualization")
	stresses.set_xlim(analysis.mesh.node_coords[:, 0].min(), analysis.mesh.node_coords[:, 0].max())
	stresses.set_ylim(analysis.mesh.node_coords[:, 1].min(), analysis.mesh.node_coords[:, 1].max())
	stresses.axes.get_xaxis().set_ticks([])
	stresses.axes.get_yaxis().set_ticks([])
	stresses.set_aspect('equal')
	plt.tight_layout()
	plt.suptitle(f"$\\epsilon$ = {analysis.eps_x * 100} [%], $\\chi_y$ = {analysis.xsi_y *1000} [mrad], $\\chi_z$ = {analysis.xsi_z*1000} [mrad]")
	plt.show()
     
def plot_stress_strain_RC(analysis):

    # Normalize strains for color mapping
    strains_percent = [x * 100 for x in analysis.strains]

    max_strain = max(abs(min(strains_percent)), abs(max(strains_percent)))
    norm = colors.TwoSlopeNorm(vmin=-max_strain, vcenter=0, vmax=max_strain)
    cmap = plt.colormaps.get_cmap('coolwarm')

    fig, (strains, stresses) = plt.subplots(1,2, figsize=(12, 4))

    for i, elem in enumerate(analysis.mesh.elements):
        x = elem.coords[:, 0]
        y = elem.coords[:, 1]
        poly = patches.Polygon(np.column_stack([x, y]),
                                edgecolor='black',
                                facecolor=cmap(norm(analysis.strains[i]*100)),
                                lw=0.3)
        strains.add_patch(poly)

    # Plotting nodes to improve visual correctness without marking them
    strains.plot(analysis.mesh.node_coords[:, 0],
            analysis.mesh.node_coords[:, 1],
            'o', markersize=0, color='black')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=strains, label=r"strain $\varepsilon$ [%]", shrink=0.7)

    strains.set_frame_on(False)
    strains.set_title("strain visualization")
    strains.set_xlim(analysis.mesh.node_coords[:, 0].min(), analysis.mesh.node_coords[:, 0].max())
    strains.set_ylim(analysis.mesh.node_coords[:, 1].min(), analysis.mesh.node_coords[:, 1].max())
    strains.axes.get_xaxis().set_ticks([])
    strains.axes.get_yaxis().set_ticks([])
    strains.set_aspect('equal')

    # Normalize stresses for color mapping
    max_rebar_stress = max(abs(min(analysis.stresses[analysis.material_groups["Rebar_B500B"]])), 
                            abs(max(analysis.stresses[analysis.material_groups["Rebar_B500B"]])))
    max_concrete_stress = max(abs(min(analysis.stresses[analysis.material_groups["Concrete_C30_37"]])),
                                abs(max(analysis.stresses[analysis.material_groups["Concrete_C30_37"]])))
    norm_rebar = colors.TwoSlopeNorm(vmin=-max_rebar_stress, vcenter=0, vmax=max_rebar_stress)
    norm_concrete = colors.TwoSlopeNorm(vmin=-max_concrete_stress, vcenter=0, vmax=max_concrete_stress)
    cmap_rebar = plt.colormaps.get_cmap('coolwarm')
    cmap_concrete = plt.colormaps.get_cmap('PiYG')

    for i, elem in enumerate(analysis.mesh.elements):
        x = elem.coords[:, 0]
        y = elem.coords[:, 1]
        if elem.material.name == "Concrete_C30_37":
            poly = patches.Polygon(np.column_stack([x, y]),
                                    edgecolor='black',
                                    facecolor=cmap_concrete(norm_concrete(analysis.stresses[i])),
                                    lw=0.0)

        elif elem.material.name == "Rebar_B500B":
            if abs(analysis.stresses[i]) >= 435:
                poly = patches.Polygon(np.column_stack([x, y]),
                                        edgecolor='black',
                                        facecolor=cmap_rebar(norm_rebar(analysis.stresses[i])),
                                        lw=0.4)
            else:
                poly = patches.Polygon(np.column_stack([x, y]),
                                        edgecolor='black',
                                        facecolor=cmap_rebar(norm_rebar(analysis.stresses[i])),
                                        lw=0)
        stresses.add_patch(poly)

    # Plotting nodes to improve visual correctness without marking them
    stresses.plot(analysis.mesh.node_coords[:, 0],
            analysis.mesh.node_coords[:, 1],
            'o', markersize=0, color='black')

    sm_rebar = cm.ScalarMappable(cmap=cmap_rebar, norm=norm_rebar)
    sm_concrete = cm.ScalarMappable(cmap=cmap_concrete, norm=norm_concrete)
    sm_rebar.set_array([])
    sm_concrete.set_array([])
    plt.colorbar(sm_rebar, ax=stresses, label=r"rebar stress $\sigma$ [MPa]", shrink=0.7)
    plt.colorbar(sm_concrete, ax=stresses, label=r"concrete stress $\sigma$ [MPa]", shrink=0.7)

    stresses.set_frame_on(False)
    stresses.set_title("stress visualization")
    stresses.set_xlim(analysis.mesh.node_coords[:, 0].min(), analysis.mesh.node_coords[:, 0].max())
    stresses.set_ylim(analysis.mesh.node_coords[:, 1].min(), analysis.mesh.node_coords[:, 1].max())
    stresses.axes.get_xaxis().set_ticks([])
    stresses.axes.get_yaxis().set_ticks([])
    stresses.set_aspect('equal')

    plt.tight_layout()
    plt.show()
	
def plot_linear_variation_eps(analysis, eps):
    Nx = []

    for strain in eps:
        Nx.append(analysis.get_section_forces(strain, 0, 0)[0])


    plt.figure(figsize=(5, 4))
    plt.plot(eps * 100, Nx, label=r"$N_{{max}} = {:.1f} kN$".format(max(Nx)))
    plt.xlabel(r'strain - $\varepsilon$ [$\%$]')
    plt.ylabel(r'normal force - $N_x$ [$kN$]')
    plt.grid()
    plt.legend()
    plt.show()
	
def plot_linear_variation_curv(analysis, curvs):
    My = []
    Mz = []

    for curv in curvs:
        My.append(analysis.get_section_forces(0, curv, 0)[1])
        Mz.append(analysis.get_section_forces(0, 0, curv)[2])


    plt.figure(figsize=(5, 4))
    plt.plot(curvs * 1000, My, label=f'$M_y$ (max = {max(My):.3} kNm)')
    plt.plot(curvs * 1000, Mz, label=f'$M_z$ (max = {max(Mz):.3} kNm)')
    plt.xlabel(r'curvature - $\chi$ [$mrad$]')
    plt.ylabel(r'Bending Moment - $M$ [$kNm$]')
    plt.grid()
    plt.legend()
    plt.show()

def plot_influence_of_N_on_M(analysis, N, My_lim, Mz_lim, symetric=True):
    My = np.linspace(My_lim[0], My_lim[1], 501)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, (my, mz) = plt.subplots(1,2, figsize=(8, 3))
    for j, target_N in enumerate(N):
        xsi = []
        M_res = []
        for i, M in enumerate(My):
            initial_guess = [0.0, 0.0, 0.0]
            result = fsolve(analysis.system_of_equations, initial_guess, args=(target_N, M, 0), full_output=1)
            if result[2] == 1:
                xsi.append(result[0][1])
                M_res.append(M)
        my.plot([x * 1000 for x in xsi], M_res,linestyle='-',  color=colors[j], lw=1)

    Mz = np.linspace(Mz_lim[0], Mz_lim[1], 501)

    for j, target_N in enumerate(N):
        xsi = []
        M_res = []
        for i, M in enumerate(Mz):
            initial_guess = [0.0, 0.0, 0.0]
            result = fsolve(analysis.system_of_equations, initial_guess, args=(target_N, 0, M), full_output=1)
            if result[2] == 1:
                xsi.append(result[0][2])
                M_res.append(M)
        mz.plot([x * 1000 for x in xsi], M_res,linestyle='-', label=f'$N_x$ = {target_N} kN', color=colors[j], lw=1)

    # Plotting the results
    my.grid(alpha=0.5)
    mz.grid(alpha=0.5)
    if symetric:
        my.set_xlim(-0.04, 0.04)
        mz.set_xlim(-0.04, 0.04)
    else:
        my.set_xlim(-0.0, 0.04)
        mz.set_xlim(-0.0, 0.04)
    my.set_xlabel(r'curvature - $\chi$ [$mrad$]')
    mz.set_xlabel(r'curvature - $\chi$ [$mrad$]')
    my.set_ylabel(r'Bending Moment - $M_y$ [$kNm$]')
    mz.set_ylabel(r'Bending Moment - $M_z$ [$kNm$]')
    mz.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_influence_of_Mz_on_My(analysis, My_lim, Mz, symetric=True):
    My = np.linspace(My_lim[0], My_lim[1], 1001)

    plt.figure(figsize=(5, 4))
    for target_Mz in Mz:
        xsi = []
        M_res = []
        for i, M in enumerate(My):
            target_My = M
            initial_guess = [0.0, 0.0, 0.0]
            result = fsolve(analysis.system_of_equations, initial_guess, args=(0, target_My, target_Mz), full_output=1)
            # Check if the solution converged
            if result[2] == 1:
                xsi.append(result[0][1])
                M_res.append(M)
        plt.plot([x * 1000 for x in xsi], M_res,linestyle='-', label=f'$M_z$ = {target_Mz} kNm')

    # Plotting the results
    plt.title('Influence of $M_z$ on $M_y$')
    plt.legend()
    plt.grid()
    if symetric:
        plt.xlim(-0.04, 0.04)
    else:
        plt.xlim(0, 0.04)
    plt.xlabel(r'curvature - $\chi$ [$mrad$]')
    plt.ylabel(r'Bending Moment - $M_y$ [$kNm$]')
    plt.tight_layout()
    plt.show()


def plot_initial_structure(structure):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for beam in structure.beam_elements:
        start_node = beam.nodes_initial[0]
        end_node = beam.nodes_initial[1]

        # Plot nodes
        ax.scatter(*start_node, color='k', marker='o')
        ax.scatter(*end_node, color='k', marker='o')

        # Plot beam line
        ax.plot([start_node[0], end_node[0]],
                [start_node[1], end_node[1]],
                [start_node[2], end_node[2]],
                color='k', linewidth=2)
        gauss = beam.gauss_points
        for eta in gauss:
            p = (eta + 1) / 2
            point = start_node + p * (end_node - start_node)
            ax.scatter(*point, color='k', marker='s',s=15)

    ax.set_box_aspect((np.ptp([0,7000]), np.ptp([-2000,2000]), np.ptp([0,3000])))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Frame Structure')
    plt.show()

def plot_displaced_structure(solver, scale=20.0):
    solver.structure.set_displaced_nodes(solver.displacements, scale)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for beam in solver.structure.beam_elements:
        start_node = beam.nodes_initial[0]
        end_node = beam.nodes_initial[1]

        # Plot nodes
        ax.scatter(*start_node, color='k', marker='o', zorder=1)
        ax.scatter(*end_node, color='k', marker='o', zorder=1)

        # Plot beam line
        ax.plot([start_node[0], end_node[0]],
                [start_node[1], end_node[1]],
                [start_node[2], end_node[2]],
                color='k', linewidth=2, zorder=1)

    for beam in solver.structure.beam_elements:
        start_node = beam.nodes_displaced[0]
        end_node = beam.nodes_displaced[1]

        # Plot nodes
        ax.scatter(*start_node, color='r', marker='o', zorder=0)
        ax.scatter(*end_node, color='r', marker='o', zorder=0)

        # Plot beam line
        ax.plot([start_node[0], end_node[0]],
                [start_node[1], end_node[1]],
                [start_node[2], end_node[2]],
                color='r', linewidth=2, zorder=0)

    ax.set_ylim([-2000, 2000])
    ax.set_box_aspect((np.ptp([0,7000]), np.ptp([-2000,2000]), np.ptp([0,3000])))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_moments(steps, section_forces, section_strains, non_linear_solver, length):
    # extract the positioning of the cross sections along the beam
	x = np.array((non_linear_solver.structure.beam_elements[0].gauss_points + 1)/2 * length) 
     
    # extract the cuvatures at each section for the given load factor steps
    # section_strains[load factor, ID of the beam element, ID of the section, [chi_z, chi_y, delta]]
	kappa_50 = np.array(section_strains[steps[0],0,:,1])  # curvature at each x (in 1/m)
	kappa_100 = np.array(section_strains[steps[1],0,:,1])  # curvature at each x (in 1/m)
	kappa_399 = np.array(section_strains[steps[2],0,:,1])  # curvature at each x (in 1/m)

	# First integration: rotation
	theta_50 = np.zeros_like(x)
	theta_100 = np.zeros_like(x)
	theta_399 = np.zeros_like(x)
	for i in range(1, len(x)):
		dx = x[i] - x[i - 1]
		theta_50[i] = theta_50[i - 1] + 0.5 * (kappa_50[i] + kappa_50[i - 1]) * dx
		theta_100[i] = theta_100[i - 1] + 0.5 * (kappa_100[i] + kappa_100[i - 1]) * dx
		theta_399[i] = theta_399[i - 1] + 0.5 * (kappa_399[i] + kappa_399[i - 1]) * dx

	# Second integration: vertical deflection
	v_50 = np.zeros_like(x)
	v_100 = np.zeros_like(x)
	v_399 = np.zeros_like(x)
	for i in range(1, len(x)):
		dx = x[i] - x[i - 1]
		v_50[i] = v_50[i - 1] + 0.5 * (theta_50[i] + theta_50[i - 1]) * dx
		v_100[i] = v_100[i - 1] + 0.5 * (theta_100[i] + theta_100[i - 1]) * dx
		v_399[i] = v_399[i - 1] + 0.5 * (theta_399[i] + theta_399[i - 1]) * dx


	fig, (curv, rot, defl, moment) = plt.subplots(1,4, figsize=(10, 4))


	# plot the results
	curv.plot(kappa_50, x, zorder=4, color="C0")
	curv.plot(kappa_100, x, zorder=3, color="C1")
	curv.plot(kappa_399, x, zorder=2, color="C2")
	curv.plot([0,0,0,0,0,0,0,0,0,0], x, zorder=5, color="k", marker="s", markersize=2)
	curv.fill_betweenx(x, kappa_50, zorder=4, color="C0", alpha=0.2)
	curv.fill_betweenx(x, kappa_100, zorder=3, color="C1", alpha=0.2)
	curv.fill_betweenx(x, kappa_399, zorder=2, color="C2", alpha=0.2)
	curv.set_xlabel("Curvature $\\kappa$ [$1/m$]")
	curv.set_ylabel("length $x$ [$m$]")
	curv.set_xlim(-min(kappa_399)*1.2, min(kappa_399)*1.2)
	curv.grid()

	rot.plot(theta_50, x, zorder=4, color="C0")
	rot.plot(theta_100, x, zorder=3, color="C1")
	rot.plot(theta_399, x, zorder=2, color="C2")
	rot.fill_betweenx(x, theta_50, zorder=4, alpha=0.2, color="C0")
	rot.fill_betweenx(x, theta_100, zorder=3, alpha=0.2, color="C1")
	rot.fill_betweenx(x, theta_399, zorder=2, alpha=0.2, color="C2")
	rot.plot([0,0,0,0,0,0,0,0,0,0], x, zorder=5, color="k", marker="s", markersize=2)
	rot.set_xlabel("Rotation $\\theta$ [$rad$]")
	rot.set_xlim(-min(theta_399)*1.2, min(theta_399)*1.2)
	rot.grid()

	defl.plot(v_50, x, zorder=4, color="C0")
	defl.plot(v_100, x, zorder=3, color="C1")
	defl.plot(v_399, x, zorder=2, color="C2")
	defl.fill_betweenx(x, v_50, zorder=4, alpha=0.2, color="C0")
	defl.fill_betweenx(x, v_100, zorder=3, alpha=0.2, color="C1")
	defl.fill_betweenx(x, v_399, zorder=2, alpha=0.2, color="C2")
	defl.plot([0,0,0,0,0,0,0,0,0,0], x, zorder=5, color="k", marker="s", markersize=2)
	defl.set_xlabel("Displacement $u$ [$mm$]")
	defl.set_xlim(-min(v_399)*1.2, min(v_399)*1.2)
	defl.grid()

    # extract the bending moments at each section for the given load factor steps
    # section_forces[load factor, ID of the beam element, ID of the section, [M_z, M_y, N]]
	My_50 = section_forces[steps[0],0,:,1] / 1000 / 1000
	My_100 = section_forces[steps[1],0,:,1] / 1000 / 1000
	My_399 = section_forces[steps[2],0,:,1] / 1000 / 1000
    
    # plot the results
	moment.plot(My_50, x, zorder=4, color="C0", label="400 kN")
	moment.plot(My_100, x, zorder=3, color="C1", label="525 kN")
	moment.plot(My_399, x, zorder=2, color="C2", label="650 kN")
	moment.fill_betweenx(x, My_50, zorder=4, alpha=0.2, color="C0")
	moment.fill_betweenx(x, My_100, zorder=3, alpha=0.2, color="C1")
	moment.fill_betweenx(x, My_399, zorder=2, alpha=0.2, color="C2")
	moment.plot([0,0,0,0,0,0,0,0,0,0], x, zorder=5, color="k", marker="s", markersize=2)
	moment.set_xlabel("Bending Moment $M_y$ [$kNm$]")
	moment.set_xlim(-min(My_399)*1.2, min(My_399)*1.2)
	moment.legend(loc="upper right")
	moment.grid(zorder=0)

	plt.tight_layout()
	plt.show()