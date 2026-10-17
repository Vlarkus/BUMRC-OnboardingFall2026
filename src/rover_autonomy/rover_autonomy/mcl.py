"""Monte Carlo localization: a particle filter over the rover's pose (x, y, yaw) in the map.

Every update moves the particles by the change in odometry (motion model), weights them by how well
the scan fits the map from each particle's pose (sensor model), and resamples when the weights get
uneven. You write the four functions above ParticleFilter; ParticleFilter ties them together.
"""

import math

import numpy as np

from rover_autonomy.geometry import wrap

NUM_PARTICLES = 1000
ALPHAS = (
    0.02,
    0.02,
    0.02,
    0.02,
)  # motion noise: turn from turning, turn from driving, drive from driving, drive from turning
SIGMA_HIT = 0.15  # m: how far a beam endpoint lands from the obstacle it hit
Z_HIT, Z_RAND = (
    0.9,
    0.1,
)  # odds a beam hit a mapped obstacle vs anything else (like an unmapped rock)
MAX_RANGE = 12.0  # m: the lidar's range
BEAM_STEP = 6  # use every 6th beam, so 60 of the 360
MIN_MOVE, MIN_TURN = 0.02, 0.02  # m, rad: skip the update while the rover is parked
IN_PLACE_TRANS = 1e-6  # m: odometry moves shorter than this count as turning in place


def sample_motion_model(particles, odom_prev, odom_now, rng):
    """Move every particle by the odometry change, with noise. Returns a new (N, 3) array.

    particles: (N, 3) array of (x, y, yaw) in the map frame.
    odom_prev, odom_now: odometry poses (x, y, yaw) in the odom frame, before and after.
    rng: a numpy Generator; draw noise with rng.normal(0.0, std, N).

    Express the change as rot1 (turn to face where the rover went), trans (drive straight) and rot2
    (turn to the final heading). Give every particle its own noisy copy, with standard deviations
        rot1:  sqrt(a1 * rot1**2 + a2 * trans**2)
        trans: sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
        rot2:  sqrt(a1 * rot2**2 + a2 * trans**2)
    and replay it from the particle's own heading. Wrap the yaws. Reference: Probabilistic Robotics,
    table 5.6. Watch out when trans is nearly zero (turning in place) and when driving backward.
    """

    # Odo decomposition
    a1, a2, a3, a4 = ALPHAS
    n = len(particles)

    dx = odom_now[0] - odom_prev[0]
    dy = odom_now[1] - odom_prev[1]
    trans = math.hypot(dx, dy)
    if trans < IN_PLACE_TRANS:
        rot1 = 0.0  # turning in place: no direction of travel, so the whole turn goes into rot2
    else:
        rot1 = float(wrap(math.atan2(dy, dx) - odom_prev[2]))
        if (
            abs(rot1) > math.pi / 2
        ):  # driving backward: face the other way, drive a negative distance
            rot1 = float(wrap(rot1 + math.pi))
            trans = -trans
    rot2 = float(wrap(odom_now[2] - odom_prev[2] - rot1))

    # Noise
    std_rot1 = math.sqrt(a1 * rot1**2 + a2 * trans**2)
    std_trans = math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
    std_rot2 = math.sqrt(a1 * rot2**2 + a2 * trans**2)
    rot1_n = rot1 + rng.normal(0.0, std_rot1, n)
    trans_n = trans + rng.normal(0.0, std_trans, n)
    rot2_n = rot2 + rng.normal(0.0, std_rot2, n)

    # Replay the noisy motion from each particle's own pose
    heading = particles[:, 2] + rot1_n
    return np.column_stack(
        (
            particles[:, 0] + trans_n * np.cos(heading),
            particles[:, 1] + trans_n * np.sin(heading),
            wrap(heading + rot2_n),
        )
    )


def scan_log_likelihood(particles, ranges, angles, grid, laser_offset):
    """How well the scan fits the map from each particle. Returns an (N,) array of log-likelihoods.

    ranges, angles: (B,) arrays of beam lengths and their angles in the laser frame. All are valid hits.
    grid: the map. grid.distance_at(x, y, outside=...) gives the distance to the nearest mapped
        obstacle at world points, for arrays of any shape.
    laser_offset: (x, y, yaw) of the laser in the rover's frame.

    For every particle and beam, find where the beam ends in the map, look up the distance d from
    that endpoint to the nearest obstacle, and score it as
        Z_HIT * N(d; 0, SIGMA_HIT) + Z_RAND / MAX_RANGE
    where N is the Gaussian density. Sum the logs of the scores over the beams. Endpoints off the map
    should get the lowest score, not the highest. Try to do it without Python loops.
    """
    # Laser pose in the map, per particle. Columns of shape (N, 1) so they broadcast against (B,) beams
    lx, ly, lyaw = laser_offset
    x, y, yaw = particles[:, 0:1], particles[:, 1:2], particles[:, 2:3]
    c, s = np.cos(yaw), np.sin(yaw)
    laser_x = x + c * lx - s * ly
    laser_y = y + s * lx + c * ly

    # Beam endpoints, shape (N, B)
    beam = yaw + lyaw + angles
    end_x = laser_x + ranges * np.cos(beam)
    end_y = laser_y + ranges * np.sin(beam)

    # Score: off the map, d = inf makes the Gaussian 0, leaving only the Z_RAND floor
    d = grid.distance_at(end_x, end_y, outside=np.inf)
    gauss = np.exp(-0.5 * (d / SIGMA_HIT) ** 2) / (SIGMA_HIT * math.sqrt(2 * math.pi))
    score = Z_HIT * gauss + Z_RAND / MAX_RANGE
    return np.log(score).sum(axis=1)


def low_variance_resample(particles, weights, rng):
    """Draw N particles with probability proportional to their weights. Returns a new (N, 3) array.

    Use low-variance (systematic) resampling: one random number r in [0, 1/N), then pointers at
    r, r + 1/N, r + 2/N, ... into the cumulative sum of the weights. Probabilistic Robotics, table 4.4.
    """
    n = len(weights)
    r = rng.uniform(0.0, 1.0 / n)
    pointers = r + np.arange(n) / n
    edges = np.cumsum(weights)
    picks = np.searchsorted(edges, pointers, side="left")
    picks = np.minimum(picks, n - 1)
    return particles[picks]


def estimate_pose(particles, weights):
    """The weighted mean pose, as a (3,) array. Average the yaws as angles: the mean of 179 degrees
    and -179 degrees is 180, not 0."""
    x = np.sum(weights * particles[:, 0])
    y = np.sum(weights * particles[:, 1])
    # Average the yaws as unit vectors, then take the angle of the mean vector
    yaw = math.atan2(np.sum(weights * np.sin(particles[:, 2])), np.sum(weights * np.cos(particles[:, 2])))
    return np.array([x, y, float(wrap(yaw))])


class ParticleFilter:
    """Ties the four functions together. Call step() on every scan."""

    def __init__(self, grid, laser_offset, seed=None):
        self.grid = grid
        self.laser_offset = laser_offset
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.weights = None
        self.odom = None

    def initialize(self, pose, std=(0.25, 0.25, 0.1)):
        """Spread the particles around a pose."""
        self.particles = np.asarray(pose, dtype=float) + self.rng.normal(
            0.0, std, (NUM_PARTICLES, 3)
        )
        self.particles[:, 2] = wrap(self.particles[:, 2])
        self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        self.odom = None

    def step(self, odom, ranges, angles):
        """Update with the odometry pose at the scan's time and the scan.

        Returns (estimate, odom_used): the pose estimate in the map, and the odometry pose it goes with.
        """
        if self.odom is not None:
            moved = math.hypot(odom[0] - self.odom[0], odom[1] - self.odom[1])
            turned = abs(float(wrap(odom[2] - self.odom[2])))
            if moved < MIN_MOVE and turned < MIN_TURN:
                return estimate_pose(self.particles, self.weights), self.odom
            self.particles = sample_motion_model(
                self.particles, self.odom, odom, self.rng
            )
        self.odom = np.asarray(odom, dtype=float)

        ranges = np.asarray(ranges, dtype=float)[::BEAM_STEP]
        angles = np.asarray(angles, dtype=float)[::BEAM_STEP]
        hit = np.isfinite(ranges) & (ranges < MAX_RANGE)
        log_w = np.log(self.weights + 1e-300) + scan_log_likelihood(
            self.particles, ranges[hit], angles[hit], self.grid, self.laser_offset
        )
        w = np.exp(log_w - log_w.max())
        self.weights = w / w.sum()

        if 1.0 / np.sum(self.weights**2) < NUM_PARTICLES / 2:
            self.particles = low_variance_resample(
                self.particles, self.weights, self.rng
            )
            self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        return estimate_pose(self.particles, self.weights), self.odom
