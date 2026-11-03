import numpy as np

def controller(control_func) -> np.ndarray:
    threshold = 0.2
    g = 9.8
    m = 0.027
    i = 1.4e-5
    d = 0.028
    dt = 1.0/48.0

    kp_acc = 0.75
    kd_acc = 1.75

    kp_acc_z = 1.5
    kd_acc_z = 2.0

    kp_ang = 7.0
    kd_ang = 5.0

    k_f = 3.16 * 1e-10

    target = [[0, 0, 0], [0, 0, 2], [0, 2, 2], [3, 2, 2], [3, 2, 0]]
    target = np.array(target)

    def calc_linear_acc(target_pos: np.ndarray, current_pos: np.ndarray, current_lin_vel: np.ndarray):
        e = target_pos - current_pos

        a = kp_acc * e - kd_acc * current_lin_vel

        a[2] = kp_acc_z * e[2] - kd_acc_z * current_lin_vel[2]

        return a

    def calc_angular_acc(target_angle: np.ndarray, current_angle: np.ndarray, current_ang_vel: np.ndarray):
        e = target_angle - current_angle

        alpha = kp_ang * e - kd_ang * current_ang_vel

        return alpha

    def calc_thrust(linear_acc: np.ndarray):
        T_x = m * linear_acc[0]
        T_y = m * linear_acc[1]
        T_z = m * (linear_acc[2] + g)

        return np.array([T_x, T_y, T_z])

    def calc_delta(ang_acc: np.ndarray):
        return ang_acc * i / (4 * d)

    hover_rpm = np.sqrt(m * g / (4 * k_f))
    pos, vel = control_func([[hover_rpm] * 4])

    vel = vel[0]
    pos = pos[0]
    vel_prev = 0.0

    beta_prev = 0.0
    theta_prev = 0.0

    for p in target:
        while np.linalg.norm(p - pos) > threshold:
            a_cur = (vel - vel_prev) / dt

            theta_cur = np.arctan2(a_cur[0], a_cur[2] + g) # XZ plane
            beta_cur = np.arctan2(a_cur[1], a_cur[2] + g) # YZ plane
            angles_cur = np.array([theta_cur, beta_cur])
            
            omega_beta_cur = (beta_cur - beta_prev) / dt
            omega_theta_cur = (theta_cur - theta_prev) / dt
            angular_vel = np.array([omega_theta_cur, omega_beta_cur])

            a_target = calc_linear_acc(p, pos, vel)
            T_target = calc_thrust(a_target)

            theta_target = np.arctan2(T_target[0], T_target[2])
            beta_target = np.arctan2(T_target[1], T_target[2])
            angles_target = np.array([theta_target, beta_target])

            alpha_target = calc_angular_acc(angles_target, angles_cur, angular_vel)

            delta_x, delta_y = calc_delta(alpha_target)

            thrust = np.linalg.norm(T_target)

            F_target = np.array(
                [
                    thrust / 4 - delta_x + delta_y,
                    thrust / 4 + delta_x + delta_y,
                    thrust / 4 + delta_x - delta_y,
                    thrust / 4 - delta_x - delta_y
                ]
            )

            r_target = np.sqrt(F_target / k_f)

            vel_prev = vel
            beta_prev = beta_cur
            theta_prev = theta_cur

            pos, vel = control_func([r_target])
            pos = pos[0]
            vel = vel[0]
