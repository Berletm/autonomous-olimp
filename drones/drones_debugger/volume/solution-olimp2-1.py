import cv2
import numpy as np
from typing import Tuple, Callable

null_action = np.zeros(3)
low_red = (0, 255, 0)
high_red = (0, 255, 255)

low_blue = (10, 255, 0)
high_blue = (255, 255, 255)

center = np.array([240.0, 240.0])

k_p = np.array([1.0, 1.0, 1.0])
action_pixels2meters = np.array([1/-84, 1/81, 1.0])

dt = 1.0 / 48.0
holding_time = 0.65
max_time = 20.0

reference_size = None
target_height = initial_height = 5.0

blue_path_len = 1.0 # meters
max_forward_action = 0.75 * blue_path_len

max_path_dist = 100.0 # pixels
dist_threshold = 24.0 # pixels
corridor_width = 20.0 # pixels
forward_dir_threshold = -0.1
backward_dir_threshold = -0.85

directions = {
    "right": np.array([ 1,  0]),
    "left":  np.array([-1,  0]),
    "down":  np.array([ 0,  1]),
    "up":    np.array([ 0, -1]),
}
cur_pos = (0, 0)

visited = {}
dir_stack = []

def detect(img, low, high) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, low, high)
    _, _, stats, centers = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    return centers[1:], stats[1:]

def hold_position(img: np.ndarray, control_func: Callable) -> np.ndarray:
    red_squares, stats = detect(img, low_red, high_red)
    nearest_square_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
    nearest_square = red_squares[nearest_square_idx]

    w, h = stats[nearest_square_idx, cv2.CC_STAT_WIDTH], stats[nearest_square_idx, cv2.CC_STAT_HEIGHT]
    size = (w + h) / 2.0

    dx, dy = (center - nearest_square) 
    dz = (target_height - initial_height * reference_size / size)

    e = np.array([dx, dy, dz]) # in pixels
    e *= action_pixels2meters # in meters
    action = k_p * e

    img = control_func([action])
    img = np.array(img[0])[..., :3]

    return img

def hold_square(img: np.ndarray, control_func: Callable) -> np.ndarray:
    start_step = control_func.step_num
    while (control_func.step_num - start_step)*dt < holding_time:
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

    dz = 0.0
    while True:
        e = np.append(center - target_square, dz)
        action = k_p * e * action_pixels2meters

        if np.linalg.norm(e) < dist_threshold:
            break

        img = np.asarray(control_func([action])[0])[..., :3]

        red_squares, stats = detect(img, low_red, high_red)
        target_square_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
        target_square = red_squares[target_square_idx]
    
        w, h = stats[target_square_idx, cv2.CC_STAT_WIDTH], stats[target_square_idx, cv2.CC_STAT_HEIGHT] 
        size = (w + h) / 2.0
        dz = target_height - initial_height * (reference_size / size)

    return img    

def choose_path(blue_paths: np.ndarray) -> np.ndarray:
    x, y = cur_pos

    local_paths = [p for p in blue_paths if np.linalg.norm(p - center) < max_path_dist and 
                    any([is_on_path(p - center, d) for d in directions.values()])]
    current_dirs = [p - center for p in local_paths]

    possible_pos = []
    for d in current_dirs:
        direction = max(directions, key=lambda x: np.dot(d / np.linalg.norm(d), directions[x]))
        dx, dy = directions[direction]
        possible_pos.append((x + dx, y + dy))

    for idx, pos in enumerate(possible_pos):
        if visited.get(pos, False) is False:
            return local_paths[idx]

    prev_dir = dir_stack[-1]
    backward_dir_idx = max(list(range(len(local_paths))), key=lambda x: np.dot(-prev_dir, current_dirs[x] / np.linalg.norm(current_dirs[x])))

    return local_paths[backward_dir_idx]

def choose_direction(img: np.ndarray) -> np.ndarray:
    blue_paths, _ = detect(img, low_blue, high_blue)
    
    target_path = choose_path(blue_paths)

    return target_path, (target_path - center) / np.linalg.norm(target_path - center)

def is_returning(dir_vec):
    return bool(dir_stack) and np.dot(dir_stack[-1], dir_vec) < backward_dir_threshold

def move_forward(img: np.ndarray, control_func: Callable) -> np.ndarray:
    global dir_stack, cur_pos

    target_path, unit_dir = choose_direction(img)

    while True:
        red_squares, _ = detect(img, low_red, high_red)
        ahead_squares = [s for s in red_squares if is_ahead(s - center, unit_dir) and is_on_path(s - center, unit_dir)]

        if len(ahead_squares) > 0:
            break

        blue_paths, _ = detect(img, low_blue, high_blue)
        ahead_paths = [p for p in blue_paths if is_ahead(p - center, unit_dir) and is_on_path(p - center, unit_dir)]
        target_path = min(ahead_paths, key=lambda x: np.linalg.norm(x - center))

        e = np.append(center - target_path, 0.0)

        raw_action = k_p * action_pixels2meters * e
        unit_action = raw_action / np.linalg.norm(raw_action)

        action = max_forward_action * unit_action

        img = control_func([action])
        img = np.asarray(img[0])[..., :3]

    img = approach_square(ahead_squares, control_func)

    direction = max(directions, key=lambda x: np.dot(unit_dir, directions[x]))
    dx, dy = directions[direction]
    x, y = cur_pos
    cur_pos = (x + dx, y + dy)

    if is_returning(unit_dir):
        dir_stack.pop()
    else:
        dir_stack.append(unit_dir)

    return img


def controller(control_drones):
    global reference_size, cur_pos

    img = control_drones([null_action])
    img = np.asarray(img[0])[..., :3]

    red_squares, red_squares_stats = detect(img, low_red, high_red)
    nearest_square_idx = min(list(range(len(red_squares))), key=lambda x: np.linalg.norm(red_squares[x] - center))
    w, h = red_squares_stats[nearest_square_idx, cv2.CC_STAT_WIDTH], red_squares_stats[nearest_square_idx, cv2.CC_STAT_HEIGHT]
    reference_size = (w + h) / 2.0

    start = control_drones.step_num

    while (control_drones.step_num - start) * dt <= max_time * 0.95:
        if visited.get(cur_pos, False) is False:
            img = hold_square(img, control_drones)
        visited[cur_pos] = True
        img = move_forward(img, control_drones)