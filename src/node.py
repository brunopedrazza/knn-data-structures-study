
class Node:
    def __init__(self, point, class_, distance=None):
        self.point = point
        self.class_ = class_
        self.distance = distance

    def __str__(self):
        return '%s' % self.point + (str(self.class_) if self.class_ else "")
    