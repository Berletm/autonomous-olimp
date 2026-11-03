import numpy as np
from typing import Callable

def controller(control_func: Callable) -> None:
    targets = [np.array([0, 0, 0.1]), 
               np.array([0, 0, 2]), 
               np.array([0, 0, 0.1])]
    g = 9.8
    m = 0.027

    k_e = 1.5
    k_d = 1.0
    k_f = 3.16 * 1e-10

    # a = k_e * (z_target - z) - k_d * v_z

    pos, vel = control_func([[0.0, 0.0, 0.0, 0.0]])
    for p in targets:
        while np.linalg.norm(p - pos) > 0.2:
            z_target = p[2]
            z_cur = pos[0, 2]
            v_cur = vel[0, 2]

            a = k_e * (z_target - z_cur) - k_d * v_cur # acceleration to reach point and stop near
            T = m * (g + a) # lifting force
            r = (T / (4 * k_f))**0.5 # rotations per minute
            # T = k_f * r**2

            pos, vel = control_func([[r] * 4])
