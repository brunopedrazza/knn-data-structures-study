import numpy as np

from heap import MaxHeap
from utils import euclidean_distance
    
class KdTree:

    def __init__(self, X, X_idx, depth=0):
        if not hasattr(X, "dtype"):
            X = np.array(X)
        if not hasattr(X_idx, "dtype"):
            X_idx = np.array(X_idx)
        
        if X.shape[0] != len(X_idx):
            raise ValueError("X must have the same size of y")
        
        axis = depth % X.shape[1]

        idx_array = np.argsort(X[:, axis])
        X = X[idx_array]
        X_idx = X_idx[idx_array]

        n = X.shape[0]
        mid = n // 2

        self.point = X[mid]
        self.idx = X_idx[mid]
        self.left = self.right = self.parent = None

        if mid > 0:
            self.left = KdTree(X[:mid], X_idx[:mid], depth + 1)
            self.left.parent = self
        
        if n - (mid + 1) > 0:
            self.right = KdTree(X[mid+1:], X_idx[mid+1:], depth + 1)
            self.right.parent = self
    
    def __str__(self):
        return '%s' % self.point
    
    @staticmethod
    def construct(X):
        return KdTree(X, [i for i in range(0, len(X))])

    @staticmethod
    def __predict(current, target, mh: MaxHeap, depth=0):
        if not current:
            return mh
        
        # get the distance squared to save 1 unecessary calculation
        # all distance are going to be compared squared
        d_squared = euclidean_distance(current.point, target, squared=True)
        mh.add([d_squared, current.idx])

        axis = depth % len(current.point)
        if target[axis] < current.point[axis]:
            good, bad = current.left, current.right
        else:
            good, bad = current.right, current.left
        
        mh = KdTree.__predict(good, target, mh, depth+1)
        r_ = target[axis] - current.point[axis]

        # if the most distant point to the target is further than the current point on axis,
        # we need to check the bad side

        # we are comparing with the square of r_ because our distances are squared
        if mh.heap[0][0] >= r_ * r_ or not mh.is_full():
            mh = KdTree.__predict(bad, target, mh, depth+1)

        return mh
    
    def predict(self, X, k):
        best_idxs = np.empty((X.shape[0], k, 2), dtype=np.int32)
        for i, x in enumerate(X):
            mh = self.__predict(self, x, MaxHeap(k=k))
            # print(mh.count)
            best_idxs[i] = mh.heap

        indices = np.argsort(best_idxs[:,:,0], axis=1)
        return best_idxs[np.arange(best_idxs.shape[0])[:, None], indices, 1]

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
    
