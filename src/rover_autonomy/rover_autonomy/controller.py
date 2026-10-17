"""MPPI (model predictive path integral) control: follow the path fast without hitting anything.

Every control step: add noise to the current plan of commands to get many candidate plans, simulate
each one, score the trajectories, and average the candidates, weighting the good ones heavily.
You write rollout(), trajectory_cost() and mppi_weights(); MPPI.step() ties them together.
"""
import numpy as np
from scipy.spatial import cKDTree

SAMPLES = 800                   # candidate plans per step
HORIZON = 30                    # steps per plan
DT = 0.1                        # s per step, so plans look 3 s ahead
V_MAX, W_MAX = 1.5, 3.0         # m/s, rad/s: the rover's speed limits
ACCEL_V, ACCEL_W = 1.0, 4.0     # m/s^2, rad/s^2: how fast the rover can change speed
NOISE = (0.4, 1.0)              # std of the noise added to v and w
TEMPERATURE = 2.0               # lower: trust the best few candidates; higher: average more of them
ROBOT_RADIUS = 0.3              # m: closer than this to an obstacle is a collision
SAFE_DISTANCE = 0.6             # m: closer than this costs extra
MAX_PATH_DEVIATION = 1.2        # m: progress stops counting this far off the path
W_PROGRESS, W_PATH, W_OBSTACLE, W_COLLISION = 12.0, 4.0, 200.0, 1000.0


def rollout(state, controls):
    """Simulate every candidate plan. Returns (K, T, 3) poses: (x, y, yaw) after each step.

    state: (x, y, yaw, v, w), where the rover is and how fast it's going right now.
    controls: (K, T, 2) commanded (v, w) for K plans of T steps.

    Each step of DT: move v toward the commanded v by at most ACCEL_V * DT (and w the same way with
    ACCEL_W), then yaw += w * DT, x += v * cos(yaw) * DT, y += v * sin(yaw) * DT. Simulate all K plans
    at once with numpy; only loop over the T steps.
    """
    k, t_steps = controls.shape[:2]
    x, y, yaw, v, w = (np.full(k, float(value)) for value in state)
    poses = np.empty((k, t_steps, 3))
    for t in range(t_steps):
        v = v + np.clip(controls[:, t, 0] - v, -ACCEL_V * DT, ACCEL_V * DT)
        w = w + np.clip(controls[:, t, 1] - w, -ACCEL_W * DT, ACCEL_W * DT)
        yaw = yaw + w * DT
        x = x + v * np.cos(yaw) * DT
        y = y + v * np.sin(yaw) * DT
        poses[:, t] = np.stack([x, y, yaw], axis=1)
    return poses


def trajectory_cost(poses, path_lookup, obstacle_distance):
    """Score every trajectory; lower is better. Returns a (K,) array.

    poses: (K, T, 3) from rollout().
    path_lookup(points): for (P, 2) points, returns two (P,) arrays: the distance to the path, and how
        far along the path (in meters, from the rover) the nearest path point is.
    obstacle_distance(points): (P,) distance to the nearest obstacle, mapped or seen in the scan.

    Reward progress along the path, and penalize distance from the path, getting closer than
    SAFE_DISTANCE to obstacles, and collisions (closer than ROBOT_RADIUS). Progress should stop counting
    once a trajectory collides or strays more than MAX_PATH_DEVIATION from the path; otherwise cutting
    a corner of the slalom looks like progress. The W_ constants are a starting point for the weights.
    """
    k, t_steps = poses.shape[:2]
    points = poses[:, :, :2].reshape(-1, 2)
    off_path, along = (a.reshape(k, t_steps) for a in path_lookup(points))
    clearance = obstacle_distance(points).reshape(k, t_steps)

    collided = clearance < ROBOT_RADIUS
    # Progress counts only up to the first step that collides or strays too far from the path
    still_valid = np.cumsum(collided | (off_path > MAX_PATH_DEVIATION), axis=1) == 0
    progress = np.where(still_valid, along, 0.0).max(axis=1)

    return (-W_PROGRESS * progress
            + W_PATH * off_path.mean(axis=1)
            + W_OBSTACLE * np.clip(SAFE_DISTANCE - clearance, 0.0, None).mean(axis=1)
            + W_COLLISION * collided.any(axis=1))


def mppi_weights(costs):
    """Weight for each candidate: exp(-(cost - lowest cost) / TEMPERATURE), normalized to sum to 1.
    Subtracting the lowest cost first keeps exp() from underflowing to all zeros."""
    w = np.exp(-(costs - costs.min()) / TEMPERATURE)
    return w / w.sum()


class MPPI:
    """Ties the three functions together. Call step() at every control tick."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.plan = np.zeros((HORIZON, 2))

    def step(self, state, path_lookup, obstacle_distance):
        """Returns (command, planned trajectory, sampled trajectories)."""
        controls = self.plan + self.rng.normal(0.0, 1.0, (SAMPLES, HORIZON, 2)) * NOISE
        controls[0] = self.plan  # always consider the current plan unchanged
        controls[1] = 0.0        # and stopping
        controls[:, :, 0] = np.clip(controls[:, :, 0], 0.0, V_MAX)
        controls[:, :, 1] = np.clip(controls[:, :, 1], -W_MAX, W_MAX)

        poses = rollout(state, controls)
        weights = mppi_weights(trajectory_cost(poses, path_lookup, obstacle_distance))
        self.plan = np.tensordot(weights, controls, axes=1)

        command = self.plan[0].copy()
        planned = rollout(state, self.plan[None])[0]
        self.plan = np.vstack([self.plan[1:], self.plan[-1:]])  # next step starts from the rest of this plan
        return command, planned, poses


class PathTracker:
    """Tracks the rover's progress along the global path and answers path_lookup queries."""

    def __init__(self, path, window=8.0):
        self.path = np.asarray(path, dtype=float)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))])
        self.window = window
        self.index = None

    def lookup_ahead(self, position):
        """A path_lookup function for the next `window` meters of path past the rover."""
        # Search near the last match so the rover can't jump to a later stretch of path that passes close by.
        lo, hi = (0, len(self.path)) if self.index is None else (max(0, self.index - 20), self.index + 100)
        self.index = lo + int(np.argmin(np.linalg.norm(self.path[lo:hi] - position, axis=1)))
        end = int(np.searchsorted(self.s, self.s[self.index] + self.window, side='right'))
        tree = cKDTree(self.path[self.index:end])
        along = self.s[self.index:end] - self.s[self.index]

        def path_lookup(points):
            distance, nearest = tree.query(points)
            return distance, along[nearest]
        return path_lookup

    def remaining(self):
        return self.s[-1] - self.s[self.index or 0]


def obstacle_distance_function(grid, scan_points):
    """obstacle_distance(points): distance to the nearest mapped obstacle or scan hit (an (N, 2) array)."""
    tree = cKDTree(scan_points) if len(scan_points) else None

    def obstacle_distance(points):
        d = grid.distance_at(points[:, 0], points[:, 1], outside=0.0)
        if tree is not None:
            d = np.minimum(d, tree.query(points, distance_upper_bound=1.0)[0])
        return d
    return obstacle_distance
