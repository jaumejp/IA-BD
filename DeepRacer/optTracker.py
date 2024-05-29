import math
import numpy as np
import matplotlib.pyplot as plt

def distance(p, q):
    return np.sqrt((p[0] -q[0])**2 +(p[1] -q[1])**2)

def inrange(h):
    return h if np.abs(h) < np.pi else (h +2 *np.pi if h < 0 else h -2 *np.pi)

def crossing(trace, outer_waypoint, inner_waypoint, waypoint):

    trace_p, trace_h, _ = trace

    # y = tan(trace_h) *x + trace_b for (x, y) = trace_p
    trace_b = trace_p[1] -math.tan(trace_h) *trace_p[0]
    # waypoint traversal heading
    waypoint_h = math.atan2((outer_waypoint[1] -inner_waypoint[1]), (outer_waypoint[0] -inner_waypoint[0]))
    
    # y = tan(waypoint_h) *x + waypoint_b
    # for e.g. outer_lane[waypoint](x, y) (Att!! center_lane[waypoint] does not always belong to the traversal line)
    waypoint_b = outer_waypoint[1] -math.tan(waypoint_h) *outer_waypoint[0]
    
    # crossing point coordinates
    if (math.tan(trace_h) -math.tan(waypoint_h)):
        crossing_x = (waypoint_b - trace_b) /(math.tan(trace_h) -math.tan(waypoint_h))
    else:
        crossing_x = (trace_p[0] + (outer_waypoint[0] + inner_waypoint[0]) /2) /2
        print('+++ crossing point error for waypoint %3d' %waypoint)
    crossing_y = math.tan(waypoint_h) *crossing_x +waypoint_b
    
    return np.array([crossing_x, crossing_y])

class Cone():
    
    def __init__(self, vertex, outer_lane, inner_lane, waypoint):
                
        self.outer_lane = outer_lane
        self.inner_lane = inner_lane
        self.track_len = self.outer_lane.shape[0]
        
        self.vertex = vertex
        self.outer_end = self.outer_lane[waypoint]
        self.inner_end = self.inner_lane[waypoint]

        self.outer_heading = math.atan2((self.outer_end[1] -self.vertex[1]), (self.outer_end[0] -self.vertex[0]))
        self.inner_heading = math.atan2((self.inner_end[1] -self.vertex[1]), (self.inner_end[0] -self.vertex[0]))
        
        if self.inner_heading > self.outer_heading:
            self.inner_heading -= 2 *np.pi
        
    def close(self, waypoint, margin = 0.005):

        new_outer_end = self.outer_lane[waypoint]
        new_inner_end = self.inner_lane[waypoint]
        
        new_outer_heading = math.atan2((new_outer_end[1] -self.vertex[1]), (new_outer_end[0] -self.vertex[0]))
        new_inner_heading = math.atan2((new_inner_end[1] -self.vertex[1]), (new_inner_end[0] -self.vertex[0]))
        
        closed = False
        
        # control for headings going from positive to negative quadrants
        if (self.outer_heading <= -np.pi /2 and new_outer_heading > 0):
            new_outer_heading -= 2 *np.pi
        if (self.outer_heading >= np.pi /2 and new_outer_heading < -np.pi /2):
            new_outer_heading += 2 *np.pi
        if (self.inner_heading <= -np.pi /2 and new_inner_heading > 0):
            new_inner_heading -= 2 *np.pi
        if (self.inner_heading >= np.pi /2 and new_inner_heading < -np.pi /2):
            new_inner_heading += 2 *np.pi

        if new_inner_heading > new_outer_heading:
            new_inner_heading -= 2 *np.pi

        if self.inner_heading <= new_outer_heading <= self.outer_heading:
            self.outer_heading = new_outer_heading
            self.outer_end = self.outer_lane[waypoint]
            closed = True
        if self.inner_heading <= new_inner_heading <= self.outer_heading:
            self.inner_heading = new_inner_heading
            self.inner_end = self.inner_lane[waypoint]
            closed = True

        if not closed:

            max_heading = self.outer_heading *((1 -margin) if self.outer_heading < 0 else (1 +margin))
            if self.inner_heading <= new_outer_heading <= max_heading:
                closed = True
            
            min_heading = self.inner_heading *((1 -margin) if self.inner_heading > 0 else (1 +margin))
            if min_heading <= new_inner_heading <= self.outer_heading:
                closed = True
            
        return closed
        
    def optimize(self, waypoint, level = 1, margin = 0.005):
        
        outer_heading_list = [self.outer_heading]
        inner_heading_list = [self.inner_heading]
        outer_end_list = [self.outer_end]
        inner_end_list = [self.inner_end]
        
        while self.close(waypoint, margin = margin):
            outer_heading_list.append(self.outer_heading)
            outer_end_list.append(self.outer_end)
            inner_heading_list.append(self.inner_heading)
            inner_end_list.append(self.inner_end)
            waypoint = np.mod(waypoint +1, self.track_len)

        if level != 1:
            self.outer_heading = outer_heading_list[-level if len(outer_heading_list) > level else 0]
            self.inner_heading = inner_heading_list[-level if len(inner_heading_list) > level else 0]
            self.outer_end = outer_end_list[-level if len(outer_end_list) > level else 0]
            self.inner_end = inner_end_list[-level if len(inner_end_list) > level else 0]
        
        
class Track():

    def __init__(self, trackFileName, clockwise = False):
    
        self.name = trackFileName
        self.clockwise = clockwise

        self.waypoints = np.load('./tracks/%s' %trackFileName)
        
        self.center_lane = self.waypoints[:-1, 0:2]
        self.inner_lane = self.waypoints[:-1, 2:4]
        self.outer_lane = self.waypoints[:-1, 4:6]
        
        self.len = self.waypoints.shape[0] -1
        self.width = np.mean(np.sum((self.inner_lane - self.outer_lane)**2, axis = 1))

        self.optimal = {}
        
    def __repr__(self):
        return '+++ %s: %4d waypoints %s' %(self.name, self.len, 'clockwise' if self.clockwise else 'counterclockwise')

    def set_clockwise(self):

        self.center_lane[:, 0] *= -1
        self.outer_lane[:, 0] *= -1
        self.inner_lane[:, 0] *= -1

        if len(self.optimal):
            for waypoint, (trace_start, trace_heading, trace_length) in self.optimal.items():
                trace_start[0] *= -1
                trace_heading = (np.pi -trace_heading) if trace_heading >= 0 else -(np.pi +trace_heading)
                self.optimal[waypoint] = (trace_start, trace_heading, trace_length)

    def optimize(self, level = 1, margin = 0.005):

        if not self.clockwise: self.set_clockwise()
        
        for waypoint in range(self.len):

            if waypoint == 0:
                trace_start = self.center_lane[waypoint]
            else:
                trace_start = crossing(self.optimal[waypoint -1], self.outer_lane[waypoint], self.inner_lane[waypoint], waypoint)

            _cone = Cone(trace_start, self.outer_lane, self.inner_lane, waypoint)
            _cone.optimize(waypoint, level = level, margin = margin)
            
            outer_length = distance(trace_start, _cone.outer_end)
            inner_length = distance(trace_start, _cone.inner_end)
            
            trace_heading = _cone.inner_heading if inner_length >= outer_length else _cone.outer_heading
            trace_length = inner_length if inner_length >= outer_length else outer_length
                
            self.optimal[waypoint] = (trace_start, trace_heading, trace_length)
  
        for waypoint in range(self.len //2):

            if waypoint == 0:
                trace_start = crossing(self.optimal[self.len -1], self.outer_lane[waypoint], self.inner_lane[waypoint], waypoint)
            else:
                trace_start = crossing(self.optimal[waypoint -1], self.outer_lane[waypoint], self.inner_lane[waypoint], waypoint)

            _cone = Cone(trace_start, self.outer_lane, self.inner_lane, waypoint)
            _cone.optimize(waypoint, level = level, margin = margin)
            
            outer_length = distance(trace_start, _cone.outer_end)
            inner_length = distance(trace_start, _cone.inner_end)
            
            trace_heading = _cone.inner_heading if inner_length >= outer_length else _cone.outer_heading
            trace_length  = inner_length if inner_length >= outer_length else outer_length
                
            self.optimal[waypoint] = (trace_start, trace_heading, trace_length)

        if not self.clockwise: self.set_clockwise()

    def plot(self, a = None, b = None, annotate = True, optimal = True):
        
        self.fig, self.ax = plt.subplots(figsize = (20, 10))
        self.ax.axis('equal')
        
        if a is None: a = 0
        if b is None: b = self.len
            
        self.ax.scatter(self.inner_lane[a:b, 0], self.inner_lane[a:b, 1], s = 1, color = 'r')
        self.ax.scatter(self.outer_lane[a:b, 0], self.outer_lane[a:b, 1], s = 1, color = 'b')
        
        if annotate:
            for i, p in enumerate(self.center_lane[a:b]):
                self.ax.annotate(a +i, (p[0], p[1]))
        else:
            self.ax.scatter(self.center_lane[a:b, 0], self.center_lane[a:b, 1], s = 1, color = 'k')
        
        if optimal and len(self.optimal):
            self.ax.scatter(self.optX(), self.optY(), s = 40, c = self.optV(), cmap = 'RdYlBu_r')

    def summary(self):

        fig, axs = plt.subplots(1, 2, figsize = (12, 5))

        L = np.sum([distance(self.center_lane[i], self.center_lane[i -1]) for i in range(1, self.len)])
        D = [distance(self.optimal[i][0], self.optimal[i -1][0]) for i in range(1, self.len)]
        V = self.optV()
        S = self.optS(degrees = True)

        # speed distribution
        axs[0].hist(V, bins = 20)
        axs[0].axvline(5, color = 'r', ls = '--', lw = 0.5)
        axs[0].set_title('Speed distribution')
        # steering distribution
        axs[1].hist(S, bins = 30)
        axs[1].axvline(-30, color = 'r', ls = '--', lw = 0.5)
        axs[1].axvline(+30, color = 'r', ls = '--', lw = 0.5)
        axs[1].set_title('Steering distribution')
        # laptime
        lapTime = np.sum([d /s for d, s, in zip(D, V)])
        #
        plt.suptitle('track length %8.2f,   lap length %8.2f,   lap time %8.2f,   avrg.speed %6.2f' %(L, np.sum(D), lapTime, np.mean(V)))

        return

    def waycone(self, waypoint = 0, horizon = 5, margin = 0.005, annotate = True, verbose = False):

        self.plot(annotate = annotate)

        if not self.clockwise: self.set_clockwise()
        xSign = 1 if self.clockwise else -1
        
        if len(self.optimal):
            p = self.optimal[waypoint][0]
        else:
            p = self.center_lane[waypoint]

        _cone = Cone(p, self.outer_lane, self.inner_lane, waypoint)
        self.ax.plot(xSign *np.array([p[0], _cone.outer_end[0]]), (p[1], _cone.outer_end[1]), 'b-')
        self.ax.plot(xSign *np.array([p[0], _cone.inner_end[0]]), (p[1], _cone.inner_end[1]), 'r-')
        if verbose:
            print('%2d %3d' %(0, waypoint), 'inn. %13.10f' %_cone.inner_heading, 'out. %13.10f' %_cone.outer_heading)

        for i, next_waypoint in enumerate(range(waypoint +1, waypoint +horizon)):
            next_waypoint = np.mod(next_waypoint, self.len)
            if not _cone.close(next_waypoint, margin = margin):
                break
            self.ax.plot(xSign *np.array([p[0], _cone.outer_end[0]]), (p[1], _cone.outer_end[1]), 'b-')
            self.ax.plot(xSign *np.array([p[0], _cone.inner_end[0]]), (p[1], _cone.inner_end[1]), 'r-')
            if verbose:
                print('%2d %3d' %(i +1, next_waypoint), 'inn. %13.10f' %_cone.inner_heading, 'out. %13.10f' %_cone.outer_heading)

        if not self.clockwise: self.set_clockwise()

    def string(self, T = 1):

        print('\tX = np.array(%s)\n' %repr(np.round(self.optX(), 4)))
        print('\tY = np.array(%s)\n' %repr(np.round(self.optY(), 4)))
        print('\tH = np.array(%s)\n' %repr(np.round(self.optH(degrees = True), 4)))
        print('\tS = np.array(%s)\n' %repr(np.round(self.optS(degrees = True), 4)))
        print('\tV = np.array(%s)\n' %repr(np.round(self.optV(), 4)))

    def optX(self):
        return [p[0] for p, h, l in self.optimal.values()]
    
    def optY(self):
        return [p[1] for p, h, l in self.optimal.values()]

    def optH(self, degrees = True):
        H = [inrange(h) for p, h, l in self.optimal.values()]
        return np.rad2deg(H) if degrees else H
    
    def optS(self, degrees = True):
        H = self.optH(degrees = False)
        S = [inrange(H[i] -H[i -1]) for i in range(1, self.len)]
        return np.rad2deg(S) if degrees else S
    
    def optV(self):
        L = [l for p, h, l in self.optimal.values()]
        return L /max(L) *5
