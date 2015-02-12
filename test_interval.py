__author__ = 'mengpeng'
from unittest import TestCase
from interval import Interval


class TestInterval(TestCase):
    def test_fromlist(self):
        l = [1, 2, 3, 10, 12, 16]
        r = Interval.fromlist(l)
        print(Interval.tolist(r))