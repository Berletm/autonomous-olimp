import numpy as np
import cv2
from typing import Callable

null_action = np.zeros(3)
low_red = (0, 255, 0)
high_red = (0, 255, 255)

low_blue = (10, 255, 0)
high_blue = (255, 255, 255)

center = np.array([240.0, 240.0])

kp = np.array([1.0, 1.0, 1.0])
pixels2meters = np.array([1/-84, 1/81, 1.0])

target_h = 5.0
hold_time = 2.0
dt = 1.0 / 48.0
n_points = 6
dist_threshold = 24.0 # pixels
corridor_width = 20
forward_dir_threshold = -0.1
max_forward_shift = 1.25

ref_square_size = None
unit_direction = None

def detect(img: np.ndarray, low: tuple, high: tuple):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, low, high)

    count, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

    return centers[1:], stats[1:]

def hold_position(img: np.ndarray, control_func: Callable):
    red_squares, stats = detect(img, low_red, high_red)

    nearest_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
    width, height = stats[nearest_idx, cv2.CC_STAT_WIDTH], stats[nearest_idx, cv2.CC_STAT_HEIGHT]
    square_size = (width + height) / 2

    cx, cy = center
    sx, sy = red_squares[nearest_idx]

    kx, ky, kz = kp * pixels2meters

    dx = kx*(cx - sx)
    dy = ky*(cy - sy)
    dz = kz*(target_h - 5*(ref_square_size/square_size)) 
    # square_size > ref_square_size -> h < target_h
    # 1 > ref / real -> 5 > 5 ref / real

    img = control_func([np.array([dx, dy, dz])])

    return np.asarray(img[0])[..., :3]

def hold_square(img: np.ndarray, control_func: Callable) -> np.ndarray:
    start_step = control_func.step_num
    while (control_func.step_num - start_step)*dt < hold_time:
        img = hold_position(img, control_func)
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


def approach_square(ahead_squares: np.ndarray, control_func: Callable) -> np.ndarray:
    target_square = np.asarray(min(ahead_squares, key=lambda x: np.linalg.norm(x - center)))

    while True:
        dx, dy = center - target_square
        e = np.array([dx, dy, 0.0])
        if np.linalg.norm(e) < dist_threshold:
            break

        action = kp * e * pixels2meters
        img = np.asarray(control_func([action])[0])[..., :3]

        red_squares, _ = detect(img, low_red, high_red)        
        target_square = min(red_squares, key=lambda x: np.linalg.norm(x - center))
    
    return img

def choose_direction(img: np.ndarray) -> np.ndarray:
    global unit_direction
    blue_paths, _ = detect(img, low_blue, high_blue)

    ahead_paths = [p for p in blue_paths if is_ahead(p - center, unit_direction)]
    nearest_path_idx = min(list(range(len(ahead_paths))), key=lambda x: np.linalg.norm(ahead_paths[x] - center))

    return ahead_paths[nearest_path_idx]

def move_forward(img: np.ndarray, control_func: Callable) -> np.ndarray:
    global unit_direction
    action = None
    next_dir = None
    while True:
        px, py = choose_direction(img)
        cx, cy = center
        kx, ky, _ = kp * pixels2meters

        next_dir = np.array([px - cx, py - cy])
        unit_direction = next_dir / np.linalg.norm(next_dir)
        
        red_squares, _ = detect(img, low_red, high_red)
        ahead_squares = [s for s in red_squares if is_ahead(s - center, unit_direction) and is_on_path(s - center, unit_direction)]
        if len(ahead_squares) > 0:
            break

        dx = kx * (cx - px) 
        dy = ky * (cy - py)

        action = np.array([dx, dy, 0.0])
        action = max_forward_shift * (action / np.linalg.norm(action))

        img = np.asarray(control_func([action])[0])[..., :3]

    img = approach_square(ahead_squares, control_func)

    return img

def controller(control_func: Callable):
    global ref_square_size, unit_direction

    img = np.asarray(control_func([null_action])[0])[..., :3]
    red_squares, red_stats = detect(img, low_red, high_red)
    blue_path, _ = detect(img, low_blue, high_blue)

    nearest_blue_path = min(blue_path, key=lambda x: np.linalg.norm(x - center))
    forward_dir = (nearest_blue_path - center)
    unit_direction = forward_dir / np.linalg.norm(forward_dir)
    
    if len(red_squares) > 0:
        nearest_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
        width, height = red_stats[nearest_idx, cv2.CC_STAT_WIDTH], red_stats[nearest_idx, cv2.CC_STAT_HEIGHT]

        ref_square_size = (width + height) / 2

        for i in range(n_points):
            img = hold_square(img, control_func)
            if i != n_points - 1:
                img = move_forward(img, control_func)
            
    else:
        raise Exception(f"no red squares found on img")
