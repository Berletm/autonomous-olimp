import cv2
import numpy as np
from typing import Tuple, Callable
from collections import deque

null_action = np.zeros(3)
low_red = (0, 255, 0)
high_red = (0, 255, 255)

low_blue = (10, 255, 0)
high_blue = (255, 255, 255)

center = np.array([240.0, 240.0])

holding_k = (
    np.array([1.0, 1.0, 1.0]),
    np.array([0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1])
)

forward_k = (
    np.array([1.0, 1.0, 1.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0])
)

recover_k = (
    np.array([1.0, 1.0, 1.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0])
)

action_pixels2meters = np.array([1/-84, 1/81, 1.0])

dt = 1.0 / 48.0
holding_time = 5.5
max_time = 20.0

reference_size = None
target_height = initial_height = 5.0

blue_path_len = 1.0 # meters
max_forward_action = 0.5 * blue_path_len

dist_threshold = 24.0 # pixels
forward_dir_threshold = -0.1
corridor_width = 20.0 # pixels

eps = 1e-6

holding_controller = None
forward_controller = None
recover_controller = None

class Controller:
    def __init__(self, k: Tuple[np.ndarray], max_integral: float=1.0, max_action: float=1.0):
        self.kp, self.kd, self.ki = k
        self.prev_error = 0.0
        self.integral = 0.0
        self.max_action = max_action
        self.max_integral = max_integral

    def get_action(self, error: np.ndarray, delta_t: float=0.0, unit_action: bool=False) -> np.ndarray:
        p = self.kp * action_pixels2meters * error

        self.integral += 0.5 * (error + self.prev_error) * delta_t
        self.integral = np.clip(self.integral, -self.max_integral, +self.max_integral)
        i = self.ki * action_pixels2meters * self.integral

        derivative = ((error - self.prev_error) / (delta_t + eps)) if delta_t != 0.0 else 0.0
        d = self.kd * action_pixels2meters * derivative

        action = p + i + d

        if unit_action:
            action /= np.linalg.norm(action)
            action *= self.max_action

        self.prev_error = error

        return action

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

def detect(img, low, high) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, low, high)
    _, _, stats, centers = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    return centers[1:], stats[1:]

def hold_square(img: np.ndarray, control_func: Callable) -> np.ndarray:
    global holding_controller

    start_step = control_func.step_num
    prev_timestep = control_func.step_num

    holding_controller.reset()

    while (control_func.step_num - start_step)*dt < holding_time:

        red_squares, stats = detect(img, low_red, high_red)
        nearest_square_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
        nearest_square = red_squares[nearest_square_idx]

        w, h = stats[nearest_square_idx, cv2.CC_STAT_WIDTH], stats[nearest_square_idx, cv2.CC_STAT_HEIGHT]
        size = (w + h) / 2.0

        dx, dy = (center - nearest_square) 
        dz = (target_height - initial_height * reference_size / size)

        e = np.array([dx, dy, dz]) # in pixels

        current_timestep = control_func.step_num
        delta_t = (current_timestep - prev_timestep) * dt
        action = holding_controller.get_action(e, delta_t)
        prev_timestep = current_timestep

        img = control_func([action])
        img = np.array(img[0])[..., :3]

    holding_controller.reset()

    return img

def approach_square(squares: np.ndarray, control_func: Callable) -> np.ndarray:
    global recover_controller

    target_square = np.asarray(min(squares, key=lambda x: np.linalg.norm(x - center)))

    dz = 0.0
    prev_time = control_func.step_num
    recover_controller.reset()
    while True:
        e = np.append(center - target_square, dz)
        current_time = control_func.step_num
        delta_t = (current_time - prev_time) * dt
        action = recover_controller.get_action(e, delta_t)

        if np.linalg.norm(e) < dist_threshold:
            break

        img = np.asarray(control_func([action])[0])[..., :3]

        red_squares, stats = detect(img, low_red, high_red)
        target_square_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
        target_square = red_squares[target_square_idx]
    
        w, h = stats[target_square_idx, cv2.CC_STAT_WIDTH], stats[target_square_idx, cv2.CC_STAT_HEIGHT] 
        size = (w + h) / 2.0
        dz = target_height - initial_height * (reference_size / size)
        prev_time = current_time
    recover_controller.reset()

    return img    

def is_ahead(vec1, path_dir):
    vec1 = vec1 / np.linalg.norm(vec1)
    path_dir = path_dir / np.linalg.norm(path_dir)

    return np.dot(vec1, path_dir) > forward_dir_threshold

def is_on_path(vec1, path_dir):
    path_dir = path_dir / np.linalg.norm(path_dir)

    along = np.dot(vec1, path_dir)
    side  = abs(path_dir[0] * vec1[1] - path_dir[1] * vec1[0])

    return along > dist_threshold and side < corridor_width

def move_forward(img: np.ndarray, control_func: Callable) -> np.ndarray:
    global forward_controller

    blue_paths, _ = detect(img, low_blue, high_blue)

    target_path = min(blue_paths, key=lambda x: np.linalg.norm(x - center))
    unit_dir = (target_path - center) / np.linalg.norm(target_path - center)

    forward_controller.reset()
    prev_t = control_func.step_num

    while True:
        red_squares, _ = detect(img, low_red, high_red)
        ahead_squares = [s for s in red_squares if is_ahead(s - center, unit_dir) and is_on_path(s - center, unit_dir)]

        if len(ahead_squares) > 0:
            break

        blue_paths, _ = detect(img, low_blue, high_blue)
        ahead_paths = [p for p in blue_paths if is_ahead(p - center, unit_dir) and is_on_path(p - center, unit_dir)]
        target_path = min(ahead_paths, key=lambda x: np.linalg.norm(x - center))

        e = np.append(center - target_path, 0.0)

        current_t = control_func.step_num
        delta_t = (current_t - prev_t) * dt
        action = forward_controller.get_action(e, delta_t, True)

        img = control_func([action])
        img = np.asarray(img[0])[..., :3]

    forward_controller.reset()

    img = approach_square(ahead_squares, control_func)
    
    return img

def controller(control_drones):
    global reference_size, holding_controller, forward_controller, recover_controller

    holding_controller = Controller(holding_k)
    forward_controller = Controller(forward_k)
    recover_controller = Controller(recover_k)

    img = control_drones([null_action])
    img = np.asarray(img[0])[..., :3]

    red_squares, red_squares_stats = detect(img, low_red, high_red)
    nearest_square_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
    w, h = red_squares_stats[nearest_square_idx, cv2.CC_STAT_WIDTH], red_squares_stats[nearest_square_idx, cv2.CC_STAT_HEIGHT]
    reference_size = (w + h) / 2.0

   
    img = hold_square(img, control_drones)
    img = move_forward(img, control_drones)
    img = hold_square(img, control_drones)    