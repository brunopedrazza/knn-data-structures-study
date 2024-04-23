
class Node:
    def __init__(self, point):
        self.point = point
        self.left = None
        self.right = None
        self.parent = None
    
    def __str__(self):
        return '%s' % self.point

    def add_node(self, new_point, index = 0):
        next_index = 0 if index + 1 == len(new_point) else index + 1 
        if new_point[index] <= self.point[index]:
            if not self.left:
                self.left = Node(new_point)
                self.left.parent = self
            else:
                self.left.add_node(new_point, next_index)
        else:
            if not self.right:
                self.right = Node(new_point)
                self.right.parent = self
            else:
                self.right.add_node(new_point, next_index)

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        if self.right is None and self.left is None:
            line = '%s' % self.point
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.point
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.point
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.point
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
    
class KdTree:
    def __init__(self):
        self.root: Node = None
    
    def add_node(self, new_point):
        if not self.root:
            self.root = Node(new_point)
            return
        self.root.add_node(new_point)
    
    def display(self):
        self.root.display()
