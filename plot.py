from SpiralTorsionSpringOptimizer import SpiralTorsionSpring
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_graph(inputs):
    # x-axis = thickness
    # y-axis = arclength
    # z-axis = stiffness
    height = inputs['input_height']
    elasticity = inputs['input_elasticity']
    max_radius_pre = inputs['input_max_radius_pre']
    radius_center = inputs['input_radius_center']
    pitch_0 = inputs['input_pitch_0']
    deltatheta_opt = inputs['input_deltatheta_opt']
    torque_pre = inputs['input_torque_pre']
    safety_factor = inputs['input_safety_factor']
    stress_yield = inputs['input_stress_yield']
    max_thickness = inputs['input_max_thickness']
    nozzle_diameter = inputs['input_nozzle_diameter']
    def f(x, y):
        z = SpiralTorsionSpring.calculate_stiffness(height, elasticity, x, y)
        cons_x = (x, y)
        cons = SpiralTorsionSpring.cons_ms(
            cons_x,
            height,
            elasticity,
            max_radius_pre,
            radius_center,
            pitch_0,
            deltatheta_opt,
            torque_pre,
            safety_factor,
            stress_yield
        )
        c1 = cons[0]
        c2 = cons[1]
        c3 = cons[2]
        z[c1 < 0] = 0
        z[c2 < 0] = 0
        z[c3 < 0] = 0
        return z
    # calculate bounds:
    min_x = 2 * nozzle_diameter
    max_x = max_radius_pre
    min_y_thickness = (  # thickness that results in shortest arclength
            3 * np.sqrt(2) * np.sqrt(torque_pre / (safety_factor * height * stress_yield))
    )
    if min_y_thickness < min_x:
        min_y_thickness = min_x
    min_y = (
            elasticity * height * min_y_thickness ** 3 * deltatheta_opt
            / (2 * (safety_factor * height * stress_yield * min_y_thickness ** 2 - 6 * torque_pre))
    )
    max_y = (
            np.pi * (max_radius_pre - radius_center / (2 * min_x)) * (max_radius_pre + radius_center)
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