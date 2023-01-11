import numpy as np
import numpy.linalg as LA

class JarvisMarch:
  def __init__(self) -> None:
    self.cvx_hull = None
    self.borderline = None
      
  def fit(self, points) -> None:
    def find_next(anchor, points):
      points = np.array([point for point in points if not np.array_equal(point, anchor)])
      cursor = points[np.random.randint(0, len(points)), :]
      for point in points:
        z = np.cross(cursor - anchor, point - anchor)
        if z > 0:
          cursor = point
      return cursor
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    left_most = points[0, :]
    next_point = find_next(left_most, points)
    hull = [left_most, next_point]
    while True:
      next_point = find_next(next_point, points)
      hull.append(next_point)
      if np.array_equal(next_point, left_most):
        break
    self.cvx_hull = np.array(hull)
    
  def expand(self, min_dist: float, circle_prec: int = 50) -> None:
    full_circle = np.linspace(0, 2, circle_prec)*np.pi
    def line(p1: np.ndarray, p2: np.ndarray) -> tuple:
      if p1[0] == p2[0]:
        raise Exception('Line is vertical. Equation is invalid!')
      x1, y1 = p1
      x2, y2 = p2
      a = (y2 - y1)/(x2 - x1)
      b = y1 - a*x1
      return (a, b)

    def vnorm(l: tuple) -> tuple:
      v = np.array([l[0], -1])
      return v/LA.norm(v), -v/LA.norm(v)

    def get_arc_pos(xy: tuple, c: tuple, r: float):
      x, y = xy
      if np.arcsin((y - c[1])/r) < 0:
        return 2*np.pi - np.arccos((x - c[0])/r)
      return np.arccos((x - c[0])/r)

    def get_arc(p1, p2, c, r):
      pos1 = get_arc_pos(p1, c, r)
      pos2 = get_arc_pos(p2, c, r)
      i1 = np.sum(full_circle < pos1)
      i2 = np.sum(full_circle < pos2)
      if i2 < i1:
        arc = np.reshape(np.concatenate((full_circle[i1:-1], full_circle[:i2])), (-1, 1))
      else:
        arc = np.reshape(full_circle[i1:i2], (-1, 1))
      points = np.concatenate((arc, arc), axis = 1)
      points[:, 0] = c[0] + r*np.cos(points[:, 0])
      points[:, 1] = c[1] + r*np.sin(points[:, 1])
      return np.array(points)
    
    # Straight lines
    hull_ = []
    for i in range(self.cvx_hull.shape[0] - 1):
      l = line(self.cvx_hull[i, :], self.cvx_hull[i + 1, :])
      u, v = vnorm(l)
      u, v = u*min_dist, v*min_dist
      if np.cross(self.cvx_hull[i + 1, :] - self.cvx_hull[i, :], u) > 0:
        hull_.append(self.cvx_hull[i, :] + u)
        hull_.append(self.cvx_hull[i + 1, :] + u)
      else:
        hull_.append(self.cvx_hull[i, :] + v)
        hull_.append(self.cvx_hull[i + 1, :] + v)
    hull_ = [hull_[-1]] + hull_[:-1]
    
    # Arcs
    path = []
    i = 0
    for c in self.cvx_hull[:-1]:
      path.append(hull_[i])
      arc = get_arc(hull_[i + 1], hull_[i], c, min_dist)
      path += reversed(arc.tolist())
      path.append(hull_[i + 1])
      i += 2
      
    path += [path[0]]
    hull_ = np.array(hull_)
    self.borderline = np.array(path)