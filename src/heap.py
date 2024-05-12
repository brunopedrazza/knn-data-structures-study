
def left_pos(pos):
    return 2 * pos + 1

def right_pos(pos):
    return 2 * (pos + 1)

def parent_pos(pos):
    return int((pos - 1)/2)

class MaxHeap:
    def __init__(self, X=None, k=None):
        self.k = k
        self.heap = []
        if X is not None:
            [self.add(x) for x in X]
    
    def __str__(self):
        s = ""
        for x in self.heap:
            s += f"{x[0]} "
        return s
    
    def add(self, x):
        x = x if isinstance(x, list) else [x, None]

        h = self.heap
        c_pos = len(h)

        h.append(x)

        self.heapify_down(c_pos)

        if self.k is not None and len(h) > self.k:
            self.remove()
    
    def remove(self):
        h = self.heap
        if len(h) == 0:
            return None
        root_v = h[0]

        last = h.pop()
        if len(h) == 0:
            return last
        h[0] = last

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
        
        best_pos = self.compare_pos(pos, left_pos(pos), right_pos(pos))
        if best_pos == pos:
            return

        self.swap(best_pos, pos)
        self.heapify_up(best_pos)
    
    def heapify_down(self, pos):
        if pos == 0:
            return
        
        p_pos = parent_pos(pos)
        best_pos = self.compare_pos(pos, p_pos)
        if best_pos == pos:
            self.swap(pos, p_pos)
            self.heapify_down(p_pos)
    
    def is_leaf(self, pos):
        return not self.is_valid_pos(left_pos(pos)) and not self.is_valid_pos(right_pos(pos))

    def is_valid_pos(self, pos):
        return pos <= len(self.heap) - 1
    
    def is_full(self):
        return len(self.heap) == self.k

    def compare_pos(self, c_pos, pos1, pos2=None):
        h = self.heap
        indxs = [c_pos]
        values = [h[c_pos][0]]
        if self.is_valid_pos(pos1):
            indxs.append(pos1)
            values.append(h[pos1][0])

        if pos2 is not None and self.is_valid_pos(pos2):
            indxs.append(pos2)
            values.append(h[pos2][0])

        b_idx = values.index(max(values))
        return indxs[b_idx]
        

if __name__ == "__main__":
    h = MaxHeap(k=3)
    h.add(22)
    h.add(8)
    h.add(13)
    h.add(3)
    h.add(56)
    h.add(90)

    print(h)

    l = [22,8,13,3,56,90]
    h = MaxHeap(l, k=3)
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)
    print(h.remove())
    print(h)

    # x = [22,8,13,3,56,90]
    # h = MaxHeap(x=x, k=6)
    # print(h)
    # print(h.remove())
    # print(h)
    # print(h.remove())
    # print(h)
    # print(h.remove())
    # print(h)
    # print(h.remove())
    # print(h)
