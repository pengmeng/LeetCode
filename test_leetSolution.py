import unittest
from unittest import TestCase
from LeetSolution import LeetSolution
from TreeNode import TreeNode
from ListNode import ListNode

__author__ = 'mengpeng'


class TestLeetSolution(TestCase):
    def test_isPalindrome(self):
        input_str = "A man, a plan, a canal: Panama"
        self.assertTrue(LeetSolution().isPalindrome(input_str))

    def test_issym(self):
        self.assertTrue(LeetSolution().isSymmetric(None))
        self.assertTrue(LeetSolution().isSymmetric(TreeNode.makeTree([1, 2, 2, 3, 4, 4, 3])))
        self.assertFalse(LeetSolution().isSymmetric(TreeNode.makeTree([1, 2, 2, '#', 3, '#', 3])))
        self.assertFalse(LeetSolution().isSymmetric(TreeNode.makeTree([1, '#', 2, '#', '#', '#', 3])))
        self.assertTrue(LeetSolution().isSymmetric(TreeNode.makeTree([1])))
        self.assertFalse(LeetSolution().isSymmetric(TreeNode.makeTree([1, 2, 3])))
        self.assertFalse(LeetSolution().isSymmetric(TreeNode.makeTree([2, 3, 3, 4, 5, 5])))

    def test_isSameTree(self):
        self.assertTrue(
            LeetSolution().isSameTree(
                TreeNode.makeTree([1, 2, 2, 3, 4, 4, 3]), TreeNode.makeTree([1, 2, 2, 3, 4, 4, 3])))
        self.assertFalse(
            LeetSolution().isSameTree(
                TreeNode.makeTree([1, 2, 2, '#', 3, '#', 3]), TreeNode.makeTree([1, 2, 2, 3, '#', 3])))
        self.assertFalse(
            LeetSolution().isSameTree(
                TreeNode.makeTree([1, 2, '#', 3]), TreeNode.makeTree([1, '#', 2, '#', '#', '#', 3])))
        self.assertTrue(LeetSolution().isSameTree(None, None))
        self.assertFalse(LeetSolution().isSameTree(TreeNode.makeTree([1]), None))
        self.assertTrue(LeetSolution().isSameTree(TreeNode.makeTree([1]), TreeNode.makeTree([1])))

    def test_merge(self):
        s = LeetSolution()
        A = [0]
        s.merge(A, 0, [], 0)
        self.assertEqual([0], A)
        A = []
        s.merge(A, 0, [1], 1)
        self.assertEqual([1], A)
        A = [1]
        s.merge(A, 1, [], 0)
        self.assertEqual([1], A)

    def test_deleteDuplicates(self):
        s = LeetSolution()
        s.deleteDuplicates(ListNode.makeList([1, 1, 2, 3, 3])).show()
        s.deleteDuplicates(ListNode.makeList([1, 1, 2])).show()
        self.assertIsNone(s.deleteDuplicates(ListNode.makeList(None)))

    def test_climbStairs(self):
        s = LeetSolution()
        self.assertEqual(1, s.climbStairs(1))
        self.assertEqual(2, s.climbStairs(2))
        print(s.climbStairs(10))