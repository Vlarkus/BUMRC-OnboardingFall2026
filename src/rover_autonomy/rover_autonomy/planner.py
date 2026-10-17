"""Global planning: A* on the prior map, through the course's gates in order.

You write astar(). The rest is provided: costmap() turns the map into blocked cells and cell costs,
plan_course() runs A* through a point just before and just after each gate, and smooth() tidies the path.
"""
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

ROBOT_RADIUS = 0.28    # m: cells closer than this to an obstacle are blocked
SOFT_RADIUS = 0.9      # m: cells closer than this cost extra, so paths keep to the middle of lanes
SOFT_WEIGHT = 4.0      # cost of a cell right at ROBOT_RADIUS is 1 + SOFT_WEIGHT
GATE_OFFSET = 0.6      # m: plan through points this far before and after each gate
PLAN_RESOLUTION = 0.1  # m: A* searches a coarser copy of the 5 cm map


def astar(blocked, cost, start, goal):
    """The cheapest 8-connected path from start to goal, as a list of (row, col) cells including both
    ends, or None if the goal can't be reached.

    blocked: (H, W) bool array. Never enter a blocked cell.
    cost: (H, W) array. Moving into a cell costs step * cost[cell], where step is 1 for a straight move
        and sqrt(2) for a diagonal one.
    start, goal: (row, col) tuples.

    Keep a priority queue (heapq) of cells ordered by f = g + h: g is the cheapest known cost from the
    start, and h is the octile distance to the goal, the length of the shortest 8-connected path if
    there were no obstacles. Every cell costs at least 1, so h never overestimates the remaining cost,
    which is what guarantees A* returns the cheapest path.
    """
    def h(cell):
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)  # octile distance

    g = {start: 0.0}
    came_from = {}
    frontier = [(h(start), start)]
    while frontier:
        _, cell = heapq.heappop(frontier)
        if cell == goal:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            return path[::-1]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (cell[0] + dr, cell[1] + dc)
                if n == cell or not (0 <= n[0] < blocked.shape[0] and 0 <= n[1] < blocked.shape[1]) or blocked[n]:
                    continue
                new_g = g[cell] + math.hypot(dr, dc) * cost[n]
                if new_g < g.get(n, math.inf):
                    g[n] = new_g
                    came_from[n] = cell
                    heapq.heappush(frontier, (new_g + h(n), n))
    return None


def costmap(grid):
    """Blocked cells, and the cost of entering each cell: 1 in the open, rising near obstacles."""
    d = grid.distance
    blocked = d < ROBOT_RADIUS
    closeness = np.clip((SOFT_RADIUS - d) / (SOFT_RADIUS - ROBOT_RADIUS), 0.0, 1.0)
    return blocked, 1.0 + SOFT_WEIGHT * closeness ** 2


def plan_course(grid, start_xy, gates):
    """Plan from start_xy through every gate in order. gates: list of (x, y, heading).

    Returns an (M, 2) array of world points about 10 cm apart, or None if some gate can't be reached.
    """
    coarse = grid.downsample(round(PLAN_RESOLUTION / grid.resolution))
    blocked, cost = costmap(coarse)
    # For every cell, the nearest unblocked cell: rescues a start or goal that lands too close to a wall.
    _, nearest = distance_transform_edt(blocked, return_indices=True)

    def free_cell(x, y):
        row, col = coarse.to_cell(x, y)
        row, col = int(np.clip(row, 0, coarse.height - 1)), int(np.clip(col, 0, coarse.width - 1))
        return int(nearest[0][row, col]), int(nearest[1][row, col])

    cells = [free_cell(*start_xy)]
    for x, y, heading in gates:
        dx, dy = GATE_OFFSET * math.cos(heading), GATE_OFFSET * math.sin(heading)
        for waypoint in ((x - dx, y - dy), (x + dx, y + dy)):
            leg = astar(blocked, cost, cells[-1], free_cell(*waypoint))
            if leg is None:
                return None
            cells += leg[1:]
    rows, cols = np.array(cells).T
    return smooth(np.column_stack(coarse.to_world(rows, cols)))


def smooth(path, spacing=0.1, window=7):
    """Round off the grid's staircase with a moving average, then space the points evenly."""
    if len(path) > window:
        pad = window // 2
        padded = np.vstack([np.repeat(path[:1], pad, axis=0), path, np.repeat(path[-1:], pad, axis=0)])
        kernel = np.ones(window) / window
        path = np.column_stack([np.convolve(padded[:, i], kernel, mode='valid') for i in range(2)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if s[-1] < spacing:
        return path
    s_even = np.arange(0.0, s[-1], spacing)
    return np.column_stack([np.interp(s_even, s, path[:, 0]), np.interp(s_even, s, path[:, 1])])
