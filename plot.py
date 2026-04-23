from spiral_torsion_spring_optimizer import SpiralTorsionSpring
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from copy import deepcopy

def plot_graph(sp: SpiralTorsionSpring):
    # x-axis = thickness
    # y-axis = arclength
    # z-axis = stiffness
    def f(x, y):
        f_sp = deepcopy(sp)
        f_sp.thickness = x
        f_sp.arclength_E = y
        f_sp.calculate_stiffness()
        z = f_sp.stiffness
        cons_x = (x, y)
        c1, c2, c3 = f_sp.cons_ms(cons_x)
        z[c1 < 0] = 0
        z[c2 < 0] = 0
        z[c3 < 0] = 0
        return z
    # calculate bounds:
    min_x = 2 * sp.nozzle_diameter
    max_x = sp.max_radius_pre
    min_y_thickness = (  # thickness that results in shortest arclength
            3 * np.sqrt(2) * np.sqrt(sp.torque_pre / (sp.safety_factor * sp.height * sp.stress_yield))
    )
    if min_y_thickness < min_x:
        min_y_thickness = min_x
    min_y = (
            sp.elasticity * sp.height * min_y_thickness ** 3 * sp.deltatheta_opt
            / (2 * (sp.safety_factor * sp.height * sp.stress_yield * min_y_thickness ** 2 - 6 * sp.torque_pre))
    )
    max_y = (
            np.pi * (sp.max_radius_pre - sp.radius_center / (2 * min_x)) * (sp.max_radius_pre + sp.radius_center)
    )
    # Create space
    x_ax = np.linspace(min_x, max_x, 100)
    y_ax = np.linspace(min_y, max_y, 100)
    x_ar, y_ar = np.meshgrid(x_ax, y_ax)
    z_ar = f(x_ar, y_ar)
    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x_ar, y_ar, z_ar, vmin=z_ar.min() * 2, cmap=cm.Blues)
    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    plt.savefig("spring_plot.png", dpi=300)
    plt.close()