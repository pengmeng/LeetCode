__author__ = 'mengpeng'

import unittest
from unittest import TestCase
from MP_Sort.MergeSort import *


class TestMerge(TestCase):
    def test_merge(self):
        A = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        merge(A, 0, 4, len(A))
        self.assertEqual(list(range(1, 11)), A)

    def test_merge_sort(self):
        A = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        merge_sort(A, 0, len(A)-1)
        self.assertEqual(list(range(1, 11)), A)