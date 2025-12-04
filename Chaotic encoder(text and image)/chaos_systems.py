import numpy as np
from scipy.integrate import odeint

class ChaoticSystem:
    def __init__(self, initial_conditions, params):
        self.initial_conditions = np.array(initial_conditions, dtype=float)
        self.params = params.copy()
        self.trajectory = None
        self.t = None

    def equations(self, state, t):
        raise NotImplementedError

    def simulate(self, t_span=500.0, dt=0.005):
        t = np.arange(0.0, t_span, dt)
        traj = odeint(self.equations, self.initial_conditions, t)
        self.t = t
        self.trajectory = traj
        return t, traj

    def compute_lyapunov(self, t, trajectory, eps=1e-8, perturb_component=0):
        # perturbed initial condition
        pert_ic = self.initial_conditions.copy()
        pert_ic[perturb_component] += eps
        pert_traj = odeint(self.equations, pert_ic, t)
        diff = pert_traj - trajectory
        sep = np.linalg.norm(diff, axis=1)
        tiny = 1e-30
        safe_sep = np.maximum(sep, tiny)
        ln_sep = np.log(safe_sep)
        ftle = np.zeros_like(ln_sep)
        positive_t = t > 0
        ftle[positive_t] = (np.log(safe_sep[positive_t] / eps)) / t[positive_t]
        if not np.any(positive_t):
            ftle[:] = 0.0
        else:
            first_pos_idx = np.argmax(positive_t)
            ftle[0] = ftle[first_pos_idx]
        return sep, ln_sep, ftle

class LorenzSystem(ChaoticSystem):
    def __init__(self, initial_conditions=(1.0,1.0,1.0), params=None):
        params = params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}
        super().__init__(initial_conditions, params)

    def equations(self, state, t):
        x, y, z = state
        sigma = self.params['sigma']; rho = self.params['rho']; beta = self.params['beta']
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

class ChuaSystem(ChaoticSystem):
    def __init__(self, initial_conditions=(0.7,0.0,0.0), params=None):
        params = params or {'alpha': 15.6, 'beta': 28.0, 'a': -1.143, 'b': -0.714}
        super().__init__(initial_conditions, params)

    def equations(self, state, t):
        x, y, z = state
        alpha = self.params['alpha']; beta = self.params['beta']; a = self.params['a']; b = self.params['b']
        h = b * x + 0.5 * (a - b) * (abs(x + 1) - abs(x - 1))
        dx = alpha * (y - x - h)
        dy = x - y + z
        dz = -beta * y
        return [dx, dy, dz]

class RosslerSystem(ChaoticSystem):
    def __init__(self, initial_conditions=(1.0,1.0,1.0), params=None):
        params = params or {'a': 0.2, 'b': 0.2, 'c': 5.7}
        super().__init__(initial_conditions, params)

    def equations(self, state, t):
        x, y, z = state
        a = self.params['a']; b = self.params['b']; c = self.params['c']
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return [dx, dy, dz]

def create_system(name, initial_conditions):
    name = name.lower()
    if name.startswith('lor'):
        return LorenzSystem(initial_conditions)
    elif name.startswith('ch'):
        return ChuaSystem(initial_conditions)
    elif name.startswith('ros') or name.startswith('rö') or name.startswith('ross'):
        return RosslerSystem(initial_conditions)
    else:
        raise ValueError(f"Unknown system: {name}")
