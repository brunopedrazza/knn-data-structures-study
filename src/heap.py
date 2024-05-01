
class Heap:
    def __init__(self, version="max", x=None):
        if version not in ["min", "max"]:
            raise ValueError("Invalid version, shoulbe be max or min")
        self.version = version
        if x is None:
            self.heap = []
        else:
            self.heap = x
            self.heapify()
    
    def __str__(self):
        s = ""
        for h in self.heap:
            s += f"{h} "
        return s
    
    @staticmethod
    def left_pos(pos):
        return 2 * pos + 1
    
    @staticmethod
    def right_pos(pos):
        return 2 * (pos + 1)
    
    @staticmethod
    def parent_pos(pos):
        return int((pos - 1)/2)
    
    def add(self, value):
        h = self.heap
        c_pos = len(self.heap)

        h.append(value)

        self.heapify_down(c_pos)
    
    def remove(self):
        h = self.heap
        root_v = h[0]

        h[0] = h.pop()

        self.heapify_up(0)

        return root_v
    
    def swap(self, pos1, pos2):
        h = self.heap
        a = h[pos1]
        h[pos1] = h[pos2]
        h[pos2] = a
    
    def heapify(self):
        n = len(self.heap)
        for pos in reversed(range(n//2)):
            self.heapify_up(pos)

    def heapify_up(self, pos):
        if self.is_leaf(pos):
            return
        
        best_pos = self.compare_pos(pos, self.left_pos(pos), self.right_pos(pos))
        if best_pos == pos:
            return

        self.swap(best_pos, pos)
        self.heapify_up(best_pos)
    
    def heapify_down(self, pos):
        if pos == 0:
            return
        
        h = self.heap
        p_pos = self.parent_pos(pos)
        best_pos = self.compare_pos(pos, p_pos)
        if best_pos == pos:
            self.swap(pos, p_pos)
            self.heapify_down(p_pos)
    
    def is_leaf(self, pos):
        return not self.is_valid_pos(self.left_pos(pos)) and not self.is_valid_pos(self.right_pos(pos))

    def is_valid_pos(self, pos):
        return pos <= len(self.heap) - 1

    def compare_pos(self, c_pos, pos1, pos2=None):
        h = self.heap
        indxs = [c_pos]
        values = [h[c_pos]]
        if self.is_valid_pos(pos1):
            indxs.append(pos1)
            values.append(h[pos1])

        if pos2 is not None and self.is_valid_pos(pos2):
            indxs.append(pos2)
            values.append(h[pos2])

        v = max(values) if self.version == "max" else min(values)
        b_idx = values.index(v)
        return indxs[b_idx]
        

if __name__ == "__main__":
    h = Heap(version="max")
    h.add(22)
    h.add(8)
    h.add(13)
    h.add(3)
    h.add(56)
    h.add(90)

    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)

    x = [22,8,13,3,56,90]
    h = Heap(x=x, version="min")
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
