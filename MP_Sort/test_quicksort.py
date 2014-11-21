import unittest
from unittest import TestCase
from MP_Sort.QuickSort import *
from ListNode import ListNode

__author__ = 'mengpeng'


class TestQuicksort(TestCase):
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
        A = [2, 1, 2, 4, 5, 9, 8]
        quicksort_array(A, 0, 6)
        self.assertEqual([1, 2, 2, 4, 5, 8, 9], A)

    @unittest.skip("succ")
    def test_partition_list(self):
        A = ListNode.makeList([5, 4, 3, 2, 1, 6, 10, 7, 8, 9])
        par = partition_list(A, None)
        print(par.val)
        A.show()

    def test_quicksort_list(self):
        A = ListNode.makeList([5, 4, 3, 2, 1, 6, 10, 7, 8, 9])
        quicksort_list(A, None)
        A.show()
        A = ListNode.makeList([2, 1, 2, 4, 5, 9, 8])
        quicksort_list(A, None)
        A.show()
        A = ListNode.makeList([10])
        quicksort_list(A, None)
        A.show()