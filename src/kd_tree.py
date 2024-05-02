import numpy as np

from collections import Counter
from heap import MaxHeap
from node import Node
from utils import euclidean_distance, measure_execution_time
    
class KdTree:

    def __init__(self, X, y, depth=0):
        if not hasattr(X, "dtype"):
            X = np.array(X)
        if not hasattr(y, "dtype"):
            y = np.array(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X must have the same size of y")
        
        axis = depth % X.shape[1]

        index_array = np.argsort(X[:, axis])
        X = X[index_array]
        y = y[index_array]

        n = X.shape[0]
        mid = n // 2

        self.node = Node(X[mid], y[mid])
        self.left = self.right = self.parent = None

        if mid > 0:
            self.left = KdTree(X[:mid], y[:mid], depth + 1)
            self.left.parent = self
        
        if n - (mid + 1) > 0:
            self.right = KdTree(X[mid+1:], y[mid+1:], depth + 1)
            self.right.parent = self
    
    def __str__(self):
        return str(self.node)
    
    @staticmethod
    @measure_execution_time
    def construct(X, y):
        return KdTree(X, y)
    
    @staticmethod
    def __closest(n0, n1, target):
        if n0 is None:
            return n1
        if n1 is None:
            return n0
        
        d0 = euclidean_distance(n0.node.point, target)
        d1 = euclidean_distance(n1.node.point, target)

        return n0 if d0 < d1 else n1

    @staticmethod
    def __predict(root, point, h, depth=0):
        if not hasattr(point, "dtype"):
            point = np.array(point)
    
        if not root:
            return None, h
        
        next = None
        other = None
        p = root.node.point
        axis = depth % len(p)
        if point[axis] < p[axis]:
            next = root.left
            other = root.right
        else:
            next = root.right
            other = root.left
        
        temp, h = KdTree.__predict(next, point, h, depth + 1)
        closest = KdTree.__closest(temp, root, point)

        r = euclidean_distance(point, closest.node.point)
        closest.node.distance = r
        if temp != closest:
            h.add(closest.node)
        r_ = point[axis] - p[axis]

        if r >= r_ or not h.is_full():
            temp, h = KdTree.__predict(other, point, h, depth + 1)
            closest = KdTree.__closest(temp, closest, point)
        return closest, h
        
    @measure_execution_time
    def predict(self, X, k):
        preds = []
        for x in X:
            _, h = self.__predict(self, x, MaxHeap(k=k))
            classes_ = [n.class_ for n in h.heap]
            pred, _ = Counter(classes_).most_common(1)[0]
            preds.append(pred)
        return preds


    def dfs(self, list_r = []):
        if not self:
            return
        list_r.append(self.node)
        if self.left:
            self.left.dfs()
        if self.right:
            self.right.dfs()
        
        return list_r
    
    def bfs(self):
        list_r = []
        q = [self]

        while q:
            c = q.pop(0)
            list_r.append([c.point, c.class_])
            if c.left:
                q.append(c.left)
            if c.right:
                q.append(c.right)
        
        return list_r


    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        if self.right is None and self.left is None:
            line = str(self)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = str(self)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = str(self)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = str(self)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2
    
