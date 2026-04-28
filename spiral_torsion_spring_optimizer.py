from scipy.optimize import NonlinearConstraint, OptimizeResult, brentq, shgo
from copy import copy
import numpy as np

class SpiralTorsionSpring:
    def __init__(
            self, elasticity, stress_yield, safety_factor, height = None, thickness = None, radius_center = None,
            pitch_0 = None, pitch_R = None, number_revolutions = None, arclength_E = None, radius_pre = None,
            torque_pre = None, stiffness = None, unutilized_elasticity = None, stress_max = None, deltatheta_opt = None,
            max_radius_pre = None, radius_E = None, theta_EMD = None, deltatheta_R = None, theta_E = None,
            radius_R = None, torque_pre_max = None, theta_Eend = None, thickness_bounds = None, arclength_bounds = None,
            min_thickness = None, max_thickness = None, c1 = None, c2 = None, c3 = None, shgo_result = None
    ):
        self.elasticity = elasticity
        self.stress_yield = stress_yield
        self.safety_factor = safety_factor
        self.stress_allowed = stress_yield * safety_factor
        self.height = height
        self.thickness = thickness
        self.radius_center = radius_center
        self.pitch_0 = pitch_0
        self.pitch_R = pitch_R
        self.number_revolutions = number_revolutions
        self.arclength_E = arclength_E
        self.radius_pre = radius_pre
        self.torque_pre = torque_pre
        self.stiffness = stiffness
        self.unutilized_elasticity = unutilized_elasticity
        self.stress_max = stress_max
        self.deltatheta_opt = deltatheta_opt
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
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.res = shgo_result

    @classmethod
    def maximize_stiffness(cls, inputs: dict, opt_params: dict = None):
        # instantiate spring:
        sp = cls(
            inputs['elasticity'],
            inputs['stress_yield'],
            inputs['safety_factor']
        )
        sp.height = inputs['height']
        sp.max_radius_pre = inputs['max_radius_pre']
        sp.radius_center = inputs['radius_center']
        sp.pitch_0 = inputs['pitch_0']
        sp.deltatheta_opt = inputs['deltatheta_opt']
        sp.torque_pre = inputs['torque_pre']
        sp.min_thickness = inputs['min_thickness']
        sp.max_thickness = inputs['max_thickness']

        # calculate bounds:
        # minimum thickness:
        base_min_thickness = 0.1
        torque_pre_min_thickness = np.sqrt(6 * sp.torque_pre / (sp.height * sp.stress_allowed))
        user_min_thickness = float(sp.min_thickness) if sp.min_thickness is not None else 0.0
        sp.min_thickness = max(base_min_thickness, torque_pre_min_thickness, user_min_thickness)
        # maximum thickness:
        max_thickness = sp.max_feasible_thickness()
        if sp.max_thickness:
            max_thickness = min(sp.max_thickness, max_thickness)
        # minimum arclength:
        min_arclength_E_thickness = (
            # thickness that results in shortest arclength (derivative of min_arclength_E with respect to thickness)
            3 * np.sqrt(2 * sp.torque_pre / (sp.height * sp.stress_allowed))
        )
        if min_arclength_E_thickness < sp.min_thickness:
            min_arclength_E_thickness = sp.min_thickness
        min_arclength_E = (
            # the minimum arclength required to survive total range of motion without exceeding allowable stress
            # calculate_stress_max with expanded deltatheta_R solved for arclength_E
            sp.elasticity * sp.height * min_arclength_E_thickness ** 3 * sp.deltatheta_opt
            / (2 * (sp.height * sp.stress_allowed * min_arclength_E_thickness ** 2 - 6 * sp.torque_pre))
        )
        # maximum arclength:
        max_arclength_E = (
            # longest spring that can fit in the allotted area
            np.pi * (sp.max_radius_pre ** 2 - sp.radius_center ** 2) / (sp.min_thickness + sp.pitch_0)
        )
        sp.thickness_bounds = (sp.min_thickness, max_thickness)
        sp.arclength_bounds = (min_arclength_E, max_arclength_E)

        # validate bounds:
        if sp.thickness_bounds[0] > sp.thickness_bounds[1] or sp.arclength_bounds[0] > sp.arclength_bounds[1]:
            sp.res = OptimizeResult(
                x=None,
                success=False,
                message=f"No solution possible; bounds infeasible"
            )
            return sp

        # configure optimizer parameters:
        params = {
            'n': 256,
            'iters': 3,
            'minimizer_kwargs': {
                'method': 'SLSQP',
                'options': {'ftol': 1e-9, 'maxiter': 500}
            },
            'options': None,
            'sampling_method': 'sobol',
            'workers': 1
        }
        if opt_params:
            params.update(opt_params)

        # optimize spring:
        sp.res = shgo(                                  # type: ignore
            func=sp.obj_ms,
            bounds=(
                sp.thickness_bounds,
                sp.arclength_bounds
            ),
            constraints=NonlinearConstraint(
                fun=sp.cons_ms,
                lb=0,
                ub=np.inf
            ),
            n=params['n'],
            iters=params['iters'],
            minimizer_kwargs=params['minimizer_kwargs'],
            options=params['options'],
            sampling_method=params['sampling_method'],
            workers=params['workers']
        )

        # validate shgo result:
        if sp.res.x is None:
            sp.res.success = False
            if not sp.res.message:
                sp.res.message = "SHGO found no feasible solution"
            return sp

        # calculate remaining properties:
        sp.build_spring_ms()

        # validate constraints:
        if sp.res.success:
            if round(sp.c1, 2) < 0 or round(sp.c2, 2) < 0 or round(sp.c3, 2) < 0:
                sp.res.success = False
                sp.res.message = (
                    "SHGO reported success but the solution violates constraints"
                )

        return sp

    def obj_ms(self, x):
        temp = copy(self)
        temp.thickness = x[0]
        temp.arclength_E = (x[1])
        temp.calculate_stiffness()
        # return negative stiffness to minimizer:
        return -temp.stiffness

    def cons_ms(self, x):
        # validate variables:
        if (x[0] < self.thickness_bounds[0]
            or x[0] > self.thickness_bounds[1]
            or x[1] < self.arclength_bounds[0]
            or x[1] > self.arclength_bounds[1]
        ):
            return np.array([-1e20, -1e20, -1e20])

        # calculate constraints:
        try:
            temp = copy(self)
            temp.thickness = x[0]
            temp.arclength_E = (x[1])
            temp.calculate_radius_E()
            temp.calculate_deltatheta_R()
            temp.calculate_theta_EMD()
            temp.calculate_stress_max()
            temp.calculate_radius_pre()

            # stress constraint:
            c1 = temp.stress_allowed - temp.stress_max
            # minimum radius constraint:
            c2 = temp.radius_pre - temp.radius_E
            # maximum radius constraint:
            c3 = temp.max_radius_pre - temp.radius_pre

            g = np.array([c1, c2, c3])

            # validate results:
            if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                return np.array([-1e20, -1e20, -1e20])
            return g
        except Exception:
            return np.array([-1e20, -1e20, -1e20])

    def build_spring_ms(self):
        # store results:
        self.c1, self.c2, self.c3 = self.cons_ms(self.res.x)
        self.thickness = self.res.x[0]
        self.arclength_E = (self.res.x[1])
        self.stiffness = -self.res.fun

        # calculate remaining properties:       Dependencies (not including inputs):
        self.calculate_radius_E()               # thickness
        self.calculate_deltatheta_R()           # arclength_E, thickness
        self.calculate_theta_EMD()              # radius_E, thickness, arclength_E
        self.calculate_theta_Eend()             # radius_E, thickness, arclength_E
        self.calculate_radius_pre()             # theta_EMD, arclength_E, radius_E, thickness
        self.calculate_theta_E()                # theta_EMD, deltatheta_R
        self.calculate_radius_R()               # arclength_E, theta_E, radius_E, thickness
        self.calculate_pitch_R()                # radius_R, radius_E, theta_E
        self.calculate_number_revolutions()     # theta_E
        self.calculate_stress_max()             # thickness, deltatheta_R, arclength_E
        self.calculate_torque_pre_max()         # thickness, arclength_E

    def max_feasible_thickness(self):
        # rough bounds:
        t_lo = self.min_thickness
        t_hi = self.max_radius_pre - self.radius_center

        def lowest_arclength(t):
            # lowest possible arclength for a given thickness within stress/ROM constraints
            denom = 2 * (self.height * self.stress_allowed * t**2 - 6 * self.torque_pre)
            if denom <= 0:
                return np.inf
            return self.elasticity * self.height * t**3 * self.deltatheta_opt / denom

        def f(t):
            # calculate difference between radius_pre for this thickness/lowest_arclength and max_radius_pre
            l = lowest_arclength(t)
            if not np.isfinite(l) or l <= 0:
                return 1e20
            radius_E = self.radius_center + t / 2 + self.pitch_0
            theta_IMD = 2 * np.pi * radius_E / (t + self.pitch_0)
            theta_MD = np.sqrt(4 * np.pi * (l + radius_E * theta_IMD / 2) / (t + self.pitch_0))
            theta_pre = theta_MD - theta_IMD - self.deltatheta_opt
            if theta_pre <= 0:
                return 1e20
            return 2 * l / theta_pre - radius_E + t / 2 - self.max_radius_pre

        # create grid of samples across search space:
        t_grid = np.linspace(t_lo, t_hi, 300)
        f_grid = np.array([f(t) for t in t_grid])

        # iterate over grid and find where f(t) crosses 0 (where radius_pre surpasses max_radius_pre)
        for idx in reversed(np.where(np.diff(np.sign(f_grid)))[0]):
            if f_grid[idx] <= 0 <= f_grid[idx + 1]:
                try:
                    return brentq(f, t_grid[idx], t_grid[idx + 1], xtol=1e-8)
                except ValueError:
                    pass

        # fallback: no crossing found
        return t_hi

    def calculate_stiffness(self):
        self.stiffness = (self.elasticity * self.height * self.thickness ** 3) / (12 * self.arclength_E)

    def calculate_arclength_E(self):
        # effective arclength (length of the spring center line)
        self.arclength_E = (self.elasticity * self.height * self.thickness ** 3) / (12 * self.stiffness)

    def calculate_radius_E(self):
        # effective radius (inner radius of spring center line)
        self.radius_E = self.radius_center + self.thickness / 2 + self.pitch_0

    def calculate_theta_EMD(self):
        # theta from origin to beginning of spring (I = ineffective):
        theta_IMD = (2 * np.pi * self.radius_E) / (self.thickness + self.pitch_0)
        # arclength from origin to beginning of spring:
        arclength_IMD = np.pi * self.radius_E * (theta_IMD / (2 * np.pi))
        # total arclength from origin at MD state:
        arclength_MD = self.arclength_E + arclength_IMD
        # total theta from origin at MD state:
        theta_MD = np.sqrt((4 * np.pi * arclength_MD) / (self.thickness + self.pitch_0))
        # theta of spring at MD state (E = effective):
        self.theta_EMD = theta_MD - theta_IMD

    def calculate_theta_Eend(self):
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
        # theta of spring at pre-load state (working backwards from MD state where pitch is obvious):
        theta_pre = self.theta_EMD - self.deltatheta_opt
        # outer radius of spring at pre-load state (including thickness, not just center line):
        self.radius_pre = (2 * self.arclength_E) / theta_pre - self.radius_E + self.thickness / 2

    def calculate_deltatheta_R(self):
        # deformation necessary to achieve preload torque (stress equation solved for deformation):
        deltatheta_pre = ((2 * self.arclength_E * 6 * self.torque_pre)
                          / (self.elasticity * self.height * self.thickness ** 3))
        # total deformation from rest state to MD state:
        self.deltatheta_R = deltatheta_pre + self.deltatheta_opt

    def calculate_theta_E(self):
        # theta of effective length of spring at rest state:
        self.theta_E = self.theta_EMD - self.deltatheta_R

    def calculate_radius_R(self):
        # outer radius of spring at rest:
        self.radius_R = (2 * self.arclength_E) / self.theta_E - self.radius_E + self.thickness / 2

    def calculate_pitch_R(self):
        # pitch of spring center line at rest:
        self.pitch_R = (2 * np.pi * (self.radius_R - self.radius_E)) / self.theta_E

    def calculate_number_revolutions(self):
        # number of revolutions of spring at rest:
        self.number_revolutions = self.theta_E / (2 * np.pi)

    def calculate_stress_max(self):
        # stress on the spring at MD state (not end state!):
        self.stress_max = (self.elasticity * self.thickness * self.deltatheta_R) / (2 * self.arclength_E)

    def calculate_torque_max(self):
        # torque on spring at MD state (not end state!):
        self.torque_max = (self.stress_max * self.height * self.thickness ** 2) / 6

    def calculate_torque_pre_max(self):
        # calculate maximum torque_pre the spring can handle:
        self.torque_pre_max = (((self.height * self.thickness ** 2 * 2 * self.arclength_E * self.stress_allowed)
                - (self.elasticity * self.height * self.thickness ** 3 * self.deltatheta_opt))
                / (12 * self.arclength_E))

    def verbose(self):
        from plot import plot_graph
        print(self.res.message)
        print('')
        print('Input Variable Bounds:')
        print(f'Thickness: [{round(self.thickness_bounds[0], 2)}, {round(self.thickness_bounds[1], 2)}]')
        print(f'Arclength: [{round(self.arclength_bounds[0], 2)}, {round(self.arclength_bounds[1], 2)}]')
        if not self.res.x.all():
            return
        print('')
        print('Properties:')
        print(f'Elasticity: {round(self.elasticity, 2)}MPa')
        print(f'Yield stress: {round(self.stress_yield, 2)}MPa')
        print(f'Safety factor: {self.safety_factor}')
        print(f'Stiffness: {round(self.stiffness, 2)}Nmm/rad')
        print(f'Outer radius at preload: {round(self.radius_pre, 2)}mm')
        print(f'Arclength of spring: {round(self.arclength_E, 2)}mm')
        print(f'Range of motion: {self.deltatheta_opt}rad')
        print(f'Preload torque: {self.torque_pre}Nmm')
        print('')
        print('Constraints:')
        print(f'Stress: {round(self.c1, 2)}')
        print(f'Min Radius: {round(self.c2, 2)}')
        print(f'Max Radius: {round(self.c3, 3)}')
        print('')
        print('Physical Dimensions (output to CAD):')
        print(f'Height: {self.height}mm')
        print(f'Thickness: {round(self.thickness, 2)}mm')
        print(f'Center pad radius: {self.radius_center}mm')
        print(f'Minimum coil distance: {self.pitch_0}mm')
        print(f'Pitch @ rest: {round(self.pitch_R, 2)}mm')
        print(f'Revolutions at rest: {round(self.number_revolutions, 2)}')
        print(f'Inner spiral radius: {round(self.radius_center + self.thickness/2 + self.pitch_0)}')
        if round(self.c1, 2) > 0 and round(self.torque_pre_max, 2) > 0:
            print('')
            print(f'This spring leaves {round(self.c1, 2)}MPa of elasticity unutilized!')
            print(f'Increase preload torque to {round(self.torque_pre_max, 2)}Nmm to fully utilize elasticity.')
        plot_graph(self)

    def to_dict(self):
        return {
            "thickness": self.thickness,
            "radius_E": self.radius_E,
            "pitch_R": self.pitch_R,
            "number_revolutions": self.number_revolutions,
            "arclength_E": self.arclength_E,
            "radius_pre": self.radius_pre,
            "theta_E": self.theta_E,
            "theta_EMD": self.theta_EMD,
            "theta_Eend": self.theta_Eend,
            "stiffness": self.stiffness,
            "stress_constraint": self.c1,
            "min_radius_constraint": self.c2,
            "max_radius_constraint": self.c3,
            "torque_pre_max": self.torque_pre_max,
            "bounds": {
                "thickness": self.thickness_bounds,
                "arclength": self.arclength_bounds
            }
        }

if __name__ == '__main__':
    user_inputs = {
        'elasticity': 3100,
        'stress_yield': 85,
        'safety_factor': .8,
        'height': 12,
        'max_radius_pre': 70,
        'radius_center': 15,
        'pitch_0': 0.5,
        'deltatheta_opt': 3.14,
        'torque_pre': 2800,
        'min_thickness': None,
        'max_thickness': None
    }
    spring = SpiralTorsionSpring.maximize_stiffness(user_inputs)
    spring.verbose()
