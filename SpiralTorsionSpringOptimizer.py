from pyswarm import pso
import numpy as np

class SpiralTorsionSpring:
    def __init__(
            self, elasticity, stress_yield, height = None, thickness = None, radius_center = None, pitch_0 = None,
            pitch_R = None, number_revolutions = None, arclength_E = None, radius_pre = None, deltatheta_opt = None,
            torque_pre = None, safety_factor = None, stiffness = None, unutilized_elasticity = None, stress_max = None,
            max_radius_pre = None, radius_E = None, theta_EMD = None, deltatheta_R = None, theta_E = None,
            radius_R = None, torque_pre_max = None, theta_Eend = None, thickness_bounds = None, arclength_bounds = None,
            nozzle_diameter = None, c1 = None, c2 = None, c3 = None
    ):
        self.elasticity = elasticity
        self.stress_yield = stress_yield
        self.height = height
        self.thickness = thickness
        self.radius_center = radius_center
        self.pitch_0 = pitch_0
        self.pitch_R = pitch_R
        self.number_revolutions = number_revolutions
        self.arclength_E = arclength_E
        self.radius_pre = radius_pre
        self.deltatheta_opt = deltatheta_opt
        self.torque_pre = torque_pre
        self.safety_factor = safety_factor
        self.stiffness = stiffness
        self.unutilized_elasticity = unutilized_elasticity
        self.stress_max = stress_max
        self.max_radius_pre = max_radius_pre
        self.radius_E = radius_E
        self.theta_EMD = theta_EMD
        self.deltatheta_R = deltatheta_R
        self.theta_E = theta_E
        self.radius_R = radius_R
        self.torque_pre_max = torque_pre_max
        self.theta_Eend = theta_Eend
        self.thickness_bounds = thickness_bounds
        self.arclength_bounds = arclength_bounds
        self.nozzle_diameter = nozzle_diameter
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    @classmethod
    def maximize_stiffness(cls, inputs: dict, opt_params: dict = None, seed: int = None):
        # instantiate spring:
        sp = cls(
            inputs['elasticity'],
            inputs['stress_yield']
        )
        sp.height = inputs['height']
        sp.max_radius_pre = inputs['max_radius_pre']
        sp.radius_center = inputs['radius_center']
        sp.pitch_0 = inputs['pitch_0']
        sp.deltatheta_opt = inputs['deltatheta_opt']
        sp.torque_pre = inputs['torque_pre']
        sp.safety_factor = inputs['safety_factor']
        sp.nozzle_diameter = inputs['nozzle_diameter']
        sp.max_thickness = inputs['max_thickness']

        # calculate bounds:
        min_thickness = 2 * sp.nozzle_diameter
        if sp.max_thickness is None:
            sp.max_thickness = sp.max_radius_pre
        min_arclength_E_thickness = (
            # thickness that results in shortest arclength
            3 * np.sqrt(2) * np.sqrt(sp.torque_pre / (sp.safety_factor * sp.height * sp.stress_yield))
        )
        if min_arclength_E_thickness < min_thickness:
            min_arclength_E_thickness = min_thickness
        min_arclength_E = (
            sp.elasticity * sp.height * min_arclength_E_thickness ** 3 * sp.deltatheta_opt
            / (2 * (sp.safety_factor * sp.height * sp.stress_yield * min_arclength_E_thickness ** 2 - 6 * sp.torque_pre))
        )
        max_arclength_E = (
            np.pi * (sp.max_radius_pre - sp.radius_center / (2 * min_thickness)) * (sp.max_radius_pre + sp.radius_center)
        )
        lb = (min_thickness, min_arclength_E)
        ub = (sp.max_thickness, max_arclength_E)

        # optimize spring:
        if seed is not None:
            np.random.seed(seed)
        pso_kwargs = dict(
            swarmsize=60,
            maxiter=500,
            omega=0.729,     # Clerc constriction — prevents diversity collapse in gbest PSO
            phip=1.49445,
            phig=1.49445,
            minstep=1e-6,
            minfunc=1e-6,
            debug=False,
        )
        if opt_params:
            pso_kwargs.update(opt_params)
        xopt, fopt = pso(sp.obj_ms, lb, ub, f_ieqcons=sp.cons_ms, **pso_kwargs)

        # determine feasibility:
        if sp.c1 < 0 or sp.c2 < 0 or sp.c3 < 0:
            return None

        # calculate remaining properties:
        sp.thickness = xopt[0]
        sp.arclength_E = xopt[1]
        sp.stiffness = -fopt
        sp.calculate_radius_E()
        sp.calculate_deltatheta_R()
        sp.calculate_theta_EMD()
        sp.calculate_theta_Eend()
        sp.calculate_radius_pre()
        sp.calculate_theta_E()
        sp.calculate_radius_R()
        sp.calculate_pitch_R()
        sp.calculate_number_revolutions()
        sp.calculate_stress_max()
        sp.calculate_unutilized_elasticity()
        sp.calculate_torque_pre_max()
        sp.thickness_bounds = (min_thickness, sp.max_thickness)
        sp.arclength_bounds = (min_arclength_E, max_arclength_E)

        return sp

    def cons_ms(self, x):
        self.thickness = x[0]
        self.arclength_E= x[1]
        self.calculate_radius_E()
        self.calculate_deltatheta_R()
        self.calculate_theta_EMD()
        self.calculate_stress_max()
        self.calculate_radius_pre()
        #stress constraint:
        self.c1 = -(self.stress_max - self.safety_factor * self.stress_yield)
        #positive radius constraint:
        self.c2 = self.radius_pre - self.radius_E
        #max radius constraint:
        self.c3 = self.max_radius_pre - self.radius_pre
        return [self.c1, self.c2, self.c3]

    def obj_ms(self, x):
        self.thickness = x[0]
        self.arclength_E = x[1]
        self.calculate_stiffness()
        return -self.stiffness

    def calculate_stiffness(self):
        self.stiffness = (self.elasticity * self.height * self.thickness ** 3) / (12 * self.arclength_E)

    def calculate_arclength_E(self):
        self.arclength_E = (self.elasticity * self.height * self.thickness ** 3) / (12 * self.stiffness)

    def calculate_radius_E(self):
        self.radius_E = self.radius_center + self.thickness / 2 + self.pitch_0

    def calculate_theta_EMD(self):
        theta_IMD = (2 * np.pi * self.radius_E) / (self.thickness + self.pitch_0)
        arclength_IMD = np.pi * self.radius_E * (theta_IMD / (2 * np.pi))
        arclength_MD = self.arclength_E + arclength_IMD
        theta_MD = np.sqrt((4 * np.pi * arclength_MD) / (self.thickness + self.pitch_0))
        self.theta_EMD = theta_MD - theta_IMD

    def calculate_theta_Eend(self):
        # deformation at which the spring self-collides
        # theta from origin to beginning of spring (I = ineffective):
        theta_Iend = (2 * np.pi * self.radius_E) / self.thickness
        # arclength from origin to beginning of spring:
        arclength_Iend = np.pi * self.radius_E * (theta_Iend / (2 * np.pi))
        # total arclength from origin at end state:
        arclength_end = self.arclength_E + arclength_Iend
        # total theta from origin at end state:
        theta_end = np.sqrt((4 * np.pi * arclength_end) / self.thickness)
        # theta of spring at end state (E = effective):
        self.theta_Eend = theta_end - theta_Iend

    def calculate_radius_pre(self):
        theta_pre = self.theta_EMD - self.deltatheta_opt
        self.radius_pre = (2 * self.arclength_E) / theta_pre - self.radius_E + self.thickness / 2

    def calculate_deltatheta_R(self):
        deltatheta_pre = ((2 * self.arclength_E * 6 * self.torque_pre)
                          / (self.elasticity * self.height * self.thickness ** 3))
        self.deltatheta_R = deltatheta_pre + self.deltatheta_opt

    def calculate_theta_E(self):
        self.theta_E = self.theta_EMD - self.deltatheta_R

    def calculate_radius_R(self):
        self.radius_R = (2 * self.arclength_E) / self.theta_E - self.radius_E + self.thickness / 2

    def calculate_pitch_R(self):
        self.pitch_R = (2 * np.pi * (self.radius_R - self.radius_E)) / self.theta_E

    def calculate_number_revolutions(self):
        self.number_revolutions = self.theta_E / (2 * np.pi)

    def calculate_stress_max(self):
        self.stress_max = (self.elasticity * self.thickness * self.deltatheta_R) / (2 * self.arclength_E)

    def calculate_torque_max(self):
        self.torque_max = (self.stress_max * self.height * self.thickness ** 2) / 6

    def calculate_unutilized_elasticity(self):
        self.unutilized_elasticity = self.safety_factor * self.stress_yield - self.stress_max

    def calculate_torque_pre_max(self):
        self.torque_pre_max = (((self.height * self.thickness ** 2 * 2 * self.arclength_E * self.safety_factor * self.stress_yield)
                - (self.elasticity * self.height * self.thickness ** 3 * self.deltatheta_opt))
                / (12 * self.arclength_E))

    def verbose(self):
        rep = [
            print(''),
            print('Properties:'),
            print(f'Elasticity: {round(self.elasticity, 2)}MPa'),
            print(f'Yield stress: {round(self.stress_yield, 2)}MPa'),
            print(f'Safety factor: {self.safety_factor}'),
            print(f'Stiffness: {round(self.stiffness, 2)}Nmm/rad'),
            print(f'Outer radius at preload: {round(self.radius_pre, 2)}mm'),
            print(f'Arclength of spring: {round(self.arclength_E, 2)}mm'),
            print(f'Range of motion: {self.deltatheta_opt}rad'),
            print(f'Preload torque: {self.torque_pre}Nmm'),
            print(''),
            print('Physical Dimensions (output to CAD):'),
            print(f'Height: {self.height}mm'),
            print(f'Thickness: {round(self.thickness, 2)}mm'),
            print(f'Center pad radius: {self.radius_center}mm'),
            print(f'Minimum coil distance: {self.pitch_0}mm'),
            print(f'Pitch @ rest: {round(self.pitch_R, 2)}mm'),
            print(f'Revolutions at rest: {round(self.number_revolutions, 2)}'),
            print(f'Inner spiral radius: {round(self.radius_center + self.thickness/2 + self.pitch_0)}')
        ]
        if round(self.unutilized_elasticity) != 0:
            rep.append([
                print(''),
                print(f'This spring leaves {round(self.unutilized_elasticity, 2)}MPa of elasticity unutilized!'),
                print(f'Increase preload torque to {round(self.torque_pre_max, 2)}Nmm to fully utilize elasticity.')
            ])
        if not self.arclength_bounds is None:
            rep.insert(8, print(f'Arclength bounds: {self.arclength_bounds}'))
        if not self.thickness_bounds is None:
            position = 14
            if not self.arclength_bounds is None:
                position += 1
            rep.insert(position, print(f'Thickness bounds: {self.thickness_bounds}'))
        return rep

    def to_dict(self):
        return {
            "height": self.height,
            "thickness": self.thickness,
            "radius_center": self.radius_center,
            "pitch_R": self.pitch_R,
            "number_revolutions": self.number_revolutions,
            "arclength_E": self.arclength_E,
            "radius_pre": self.radius_pre,
            "theta_E": self.theta_E,
            "theta_EMD": self.theta_EMD,
            "theta_Eend": self.theta_Eend,
            "stiffness": self.stiffness,
            "stress_yield": self.stress_yield,
            "safety_factor": self.safety_factor,
            "unutilized_elasticity": self.unutilized_elasticity,
            "torque_pre_max": self.torque_pre_max,
            "bounds": {
                "thickness": self.thickness_bounds,
                "arclength": self.arclength_bounds
            }
        }

if __name__ == '__main__':
    from plot import plot_graph
    user_inputs = {
        'elasticity': 2600,
        'stress_yield': 85,
        'height': 4,
        'max_radius_pre': 65,
        'radius_center': 10,
        'pitch_0': 0.5,
        'deltatheta_opt': 15.71,
        'torque_pre': 0,
        'safety_factor': .75,
        'max_thickness': 3,
        'nozzle_diameter': 0.4
    }
    spring = SpiralTorsionSpring.maximize_stiffness(user_inputs)
    if spring:
        spring.verbose()
        plot_graph(spring)
