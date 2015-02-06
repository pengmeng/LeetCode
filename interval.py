__author__ = 'mengpeng'


class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __str__(self):
        return "({0}, {1})".format(self.start, self.end)