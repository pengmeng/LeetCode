from unittest import TestCase
from MP_Sort.QuickSort import *
__author__ = 'mengpeng'


class TestQuicksort_array(TestCase):

    def test_quicksort_array(self):
        A = [1]
        quicksort_array(A, 0, 0)
        self.assertEqual([1], A)
        A = [2, 1]
        quicksort_array(A, 0, 1)
        self.assertEqual([1, 2], A)
        A = [5, 4, 3, 2, 1, 6, 10, 7, 8, 9]
        quicksort_array(A, 0, 9)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], A)