from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np

class Spline:
    def __init__(self, t: list, y: list, bounds: tuple=(0, 0)):
        self.t = t
        self.y = y
        self.bc = bounds
        self.coefs = None

        self.evaluate_coefs()

    def evaluate_coefs(self):
        n = len(self.y) - 1
        h = [self.t[i + 1] - self.t[i] for i in range(n)]

        mat = np.zeros(shape=(n + 1, n + 1))
        rhs = np.zeros(shape=(n + 1))

        for i in range(1, n):
            mat[i][i - 1], mat[i][i], mat[i][i + 1] = h[i - 1], 2*(h[i-1] + h[i]), h[i]
            rhs[i] = 3 * ((self.y[i + 1] - self.y[i]) / h[i] - (self.y[i] - self.y[i - 1]) / h[i - 1])


        rhs[0], rhs[n] = self.bc[0], self.bc[1]
        mat[0][0], mat[n][n] = 1, 1

        c = np.linalg.solve(mat, rhs)

        a = self.y[:-1]
        b = [0.0] * n
        d = [0.0] * n

        for i in range(n):
            b[i] = (self.y[i + 1] - self.y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        self.coefs = np.column_stack((a, b, c[:-1], d))

    def find_interval(self, t) -> int:
        if t == self.t[-1]:
            return len(self.t) - 2
        l, r = 0, len(self.t) - 2

        while l <= r:
            mid = (l + r) // 2

            if self.t[mid] <= t < self.t[mid + 1]:
                return mid
            elif t < self.t[mid]:
                r = mid - 1
            else:
                l = mid + 1

        raise RuntimeError("can not find interval")
    
    def __call__(self, t: float, order: int=0) -> float:        
        if order >= 4:
            return 0.0
        else:
            t = np.clip(t, self.t[0], self.t[-1])

            i = self.find_interval(t) 

            if i is not None:
                a, b, c, d = self.coefs[i]
                h = t - self.t[i]

                if order == 0:
                    res = (
                        a * (h ** 0) 
                        + b * (h ** 1) 
                        + c * (h ** 2) 
                        + d * (h ** 3)
                    )
                elif order == 1:
                    res = (
                    b 
                    + 2 * c * h 
                    + 3 * d * (h ** 2)
                    )
                elif order == 2:
                    res = 2 * c + 6 * d * h
                elif order == 3:
                    res =  6 * d
                else:
                    raise RuntimeError("wrong derivative order")
                
                return res
            else:
                raise RuntimeError("can not find interval")

def map2world(i: int, j: int, grid_height: int, tile_size: float):
    return i * tile_size + tile_size / 2.0, (grid_height - 1 - j) * tile_size + tile_size / 2.0

def compute_time(traj, v_max=0.7, omega_max=1.0):
    traj = np.asarray(traj, dtype=float)
    ds = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    if np.any(ds <= 0):
        raise ValueError("neighbours points must be different")

    curvature = np.zeros(len(traj))

    for i in range(1, len(traj) - 1):
        # points of triangle 
        u = traj[i] - traj[i - 1]
        v = traj[i + 1] - traj[i]
        chord = traj[i + 1] - traj[i - 1]

        denominator = (
            np.linalg.norm(u)
            * np.linalg.norm(v)
            * np.linalg.norm(chord)
        )

        if denominator > 1e-12:
            cross = u[0] * v[1] - u[1] * v[0] # signed 2 * S (area of triangle)
            curvature[i] = 2 * abs(cross) / denominator # curvature = 1 / R of circle from triangle

    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    speed = np.full(len(traj), v_max)
    curved = curvature > 1e-8 # mask for curved segments

    speed[curved] = np.minimum(
        v_max,
        omega_max / curvature[curved],
    )

    segment_speed = np.minimum(speed[:-1], speed[1:])
    dt = ds / segment_speed

    return np.concatenate(([0.0], np.cumsum(dt)))

class Controller:
    def __init__(self, target_x: Spline, target_y: Spline, coefs: tuple):
        self.target_x = target_x
        self.target_y = target_y

        self.target_teta = lambda t: (np.arctan2(self.target_y(t, order=1), self.target_x(t, order=1)))
        self.target_v = lambda t: (self.target_x(t, order=1) ** 2 + self.target_y(t, order=1) ** 2) ** 0.5
        self.target_w = lambda t: (
            self.target_y(t, order=2) * self.target_x(t, order=1) - self.target_x(t, order=2) * self.target_y(t, order=1)
        ) / (self.target_x(t, order=1) ** 2 + self.target_y(t, order=1) ** 2)

        self.coefs = coefs

    def get_action(self, t: float, cur_pos: np.ndarray, cur_teta: np.ndarray) -> np.ndarray:
        if t >= self.target_x.t[-1]:
            return np.array([0.0, 0.0])
        
        cur_x, cur_y = cur_pos
        
        target_x = self.target_x(t)
        target_y = self.target_y(t)
        target_teta = self.target_teta(t)
        k1, k2, k3 = self.coefs

        dx = (target_x - cur_x)
        dy = (target_y - cur_y)
        e1 = np.cos(cur_teta)*dx + np.sin(cur_teta)*dy
        e2 = -np.sin(cur_teta)*dx + np.cos(cur_teta)*dy
        e3 = target_teta - cur_teta
        e3 = np.arctan2(np.sin(e3), np.cos(e3))

        target_v = self.target_v(t)
        target_w = self.target_w(t)

        v = target_v * np.cos(e3) + k1*e1

        # sin(x)/x = 1 
        # x -> 0

        if abs(e3) > 1e-6:
            w = target_w + k2*(e2 * target_v * np.sin(e3) / e3) + k3*e3
        else:
            w = target_w + k2*(e2 * target_v * 1.0) + k3*e3

        v = np.clip(v, -0.9, 0.9)
        return np.array([v, w])

class Solution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)
        self.coefs = (1.0, 5.0, 5.0)
        self.dt = 1.0 / 30.0
        self.traj_pth = "/home/throwdownyourtears/Desktop/code/ado-2026/duckietown/ado1/"
        self.env = self.generated_task["env"]
        self.duck_pos = np.array([4.5 * self.env.road_tile_size, (self.env.grid_height - 1.9) * self.env.road_tile_size])

        self.points = np.array([
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (4.5, 1),

            (4.6913, 1.0381),
            (4.8536, 1.1464),
            (4.9619, 1.3087),
            (5.0, 1.5),
            (5.0, 2.5),
            (5.0, 3.5),
            (5.0, 4.5),

            (4.9619, 4.6913),
            (4.8536, 4.8536),
            (4.6913, 4.9619),
            (4.5, 5.0),
            (4.3087, 5.0381),
            (4.1464, 5.1464),
            (4.0381, 5.3087),
            (4.0, 5.5),
            (3.9619, 5.6913),
            (3.8536, 5.8536),
            (3.6913, 5.9619),
            (3.5, 6.0),
            (2.5, 6.0),
            (1.5, 6.0),

            (1.4, 5.0),
            (1.4, 4.5)
        ], dtype=float)

        self.traj = [map2world(p[1], p[0], self.env.grid_height, self.env.road_tile_size) for p in self.points]

    def solve(self):
        x, _, z = self.env.cur_pos
        traj = np.array(self.traj)

        t = compute_time(traj, v_max=0.85)
        x_spline = Spline(t, traj[:, 0])
        y_spline = Spline(t, traj[:, 1])
        
        controller = Controller(x_spline, y_spline, self.coefs)

        done = False
        step_idx = 1
        while not done:
            t = step_idx * self.dt
            x, _, z = self.env.cur_pos
            cur_z = self.env.grid_height * self.env.road_tile_size - z
            teta = self.env.cur_angle

            dist = np.linalg.norm(np.array([x, cur_z]) - self.duck_pos)
            print(dist)
            if dist <= 0.2:
                break

            action = controller.get_action(t, (x, cur_z), teta)
            state, reward, done, info = self.env.step(action)
            self.env.render(mode="top_down")
                
            step_idx += 1


if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name="ado_map_1")
    solution = Solution(task_generator.generated_task)
    solution.solve()
