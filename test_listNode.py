import unittest
from unittest import TestCase
from ListNode import ListNode

__author__ = 'mengpeng'


class TestListNode(TestCase):

    @unittest.skip("success")
    def test_show(self):
        array = [1, 2, 3, 4, 5]
        head = ListNode.makeList(array)
        head.show()