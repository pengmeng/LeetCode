__author__ = 'mengpeng'


class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __str__(self):
        return "({0}, {1})".format(self.start, self.end)

    @staticmethod
    def fromlist(l):
        result = []
        for i in range(0, len(l) - 1, 2):
            new = Interval(l[i], l[i+1])
            result.append(new)
        return result

    @staticmethod
    def tolist(l):
        result = []
        for each in l:
            result.append(each.start)
            result.append(each.end)
        return result