import numpy as np
from typing import Callable

def controller(control_func: Callable) -> None:
    targets = [[1, 0, 2], [2, 0, 3], [3, 0, 4], [2, 0, 3], [1, 0, 2]]


    for p in targets:
        control_func([p])
        if p == [2, 0, 3]:
            control_func([p])
