
class Node:
    def __init__(self, point, class_):
        self.point = point
        self.class_ = class_

    def __str__(self):
        return '%s' % self.point + (str(self.class_) if self.class_ else "")
    