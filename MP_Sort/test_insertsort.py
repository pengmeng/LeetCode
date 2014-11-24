__author__ = 'mengpeng'

from unittest import TestCase
from MP_Sort.InsertSort import *
from ListNode import ListNode


class TestInsertsort(TestCase):
    def test_insertsort_list(self):
        A = ListNode.makeList([])
        self.assertEqual(None, insertsort_list(A))
        A = ListNode.makeList([5, 4, 3, 2, 1, 6, 10, 7, 8, 9])
        insertsort_list(A).show()
        A = ListNode.makeList([2, 1, 2, 4, 5, 9, 8])
        insertsort_list(A).show()
        A = ListNode.makeList([1])
        insertsort_list(A).show()
        A = ListNode.makeList([1, 1])
        insertsort_list(A).show()
        A = ListNode.makeList(list(range(0, 5000)))
        insertsort_list(A).show()