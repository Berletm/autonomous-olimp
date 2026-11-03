# ADO 2026 · Autonomous Robotics

My Python solutions to ADO Olympiad tasks: autonomous driving in Duckietown and drone control in simulation.

## Duckietown

[Solutions for maps 1–3](duckietown/) turn hand-picked waypoints into a smooth spline trajectory, with route adjustments for each map. Spline derivatives provide the reference heading and linear and angular velocities.

At the heart of the solution is a **nonlinear tracking controller derived using a Lyapunov function**. It expresses position errors in the robot’s own frame and couples them with the heading error to compute velocity corrections. The elegance lies in choosing these corrections so that cross terms cancel in the Lyapunov derivative, making it nonpositive for the ideal, unsaturated model. This gives the feedback law a clear stability rationale while keeping the implementation compact.

| Task | Solution |
| --- | --- |
| Map 1 | [ado1/main.py](duckietown/ado1/main.py) |
| Map 2 | [ado2/main.py](duckietown/ado2/main.py) |
| Map 3 | [ado3/main.py](duckietown/ado3/main.py) |

## Drones

The [drone solutions](drones/drones_debugger/volume/) cover two main themes:

- **Flight control** (`solution0-1.py`, `solution1-*.py`): follow waypoints, progressing from direct commands to position and attitude feedback that determines motor speeds.
- **Visual navigation** (`solution2-*.py`, `solution-olimp2-*.py`): detect red checkpoints and blue paths in HSV space, steer using image feedback, and estimate altitude from marker size. Exploration variants track visited checkpoints to guide route choices.

## Tools

Python · NumPy · OpenCV · Gym Duckietown · Docker

Docker Compose configurations are included for both simulation environments; `duckietown/hsv-choose.py` helps tune color thresholds.
