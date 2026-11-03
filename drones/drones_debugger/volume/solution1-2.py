
import numpy as np
from typing import Callable

def controller(control_func: Callable) -> None:
    targets = [np.array([2.5, 0, 3]), np.array([5, 0, 3]), np.array([2.5, 0, 3]), np.array([0, 0, 3])]
    threshold = 0.2
    g = 9.8
    m = 0.027
    i = 1.4e-5
    d = 0.028
    dt = 1.0/48.0

    k_pv = 0.5
    k_dv = 1.5

    k_pa = 0.5
    k_da = 1.0

    k_f = 3.16 * 1e-10

    pos, vel = control_func([[0.0] * 4])

    prev_vel = vel
    prev_theta = np.pi / 2
    for p in targets:
        target_x = p[0]
        target_z = p[2]
        while np.linalg.norm(p - pos) > threshold:
            cur_x = pos[0, 0]
            cur_z = pos[0, 2]

            v_x = vel[0, 0]
            v_z = vel[0, 2]

            e_x = (target_x - cur_x)
            e_z = (target_z - cur_z)

            # target
            a_x = k_pv * e_x - k_dv * v_x
            a_z = k_pv * e_z - k_dv * v_z

            T_z = m * (a_z + g)
            T_x = m * a_x

            T = np.sqrt(T_x ** 2 + T_z ** 2)

            theta_target = np.arctan2(a_z + g, a_x)

            # current 
            a_cur = (vel - prev_vel) / dt
            a_z_cur = a_cur[0, 2]
            a_x_cur = a_cur[0, 0]
            theta = np.arctan2(a_z_cur + g, a_x_cur)

            angle_error = theta_target - theta
            omega = (theta - prev_theta) / dt # angular velocity

            alpha = k_pa * angle_error - k_da * omega # target angular acceleration

            F_r = 0.5 * (T + i * alpha / d)
            r_r = np.sqrt(F_r / (2 * k_f))

            F_l = 0.5 * (T - i * alpha / d)
            r_l = np.sqrt(F_l / (2 * k_f))

            action = [0.0, 0.0, 0.0, 0.0]
            action[0] = action[3] = r_r
            action[1] = action[2] = r_l

            prev_vel = vel
            prev_theta = theta

            pos, vel = control_func([action])
