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
        self.assertEqual(89, s.climbStairs(10))

    def test_plusOne(self):
        s = LeetSolution()
        self.assertEqual([2], s.plusOne([1]))
        self.assertEqual([1, 0], s.plusOne([9]))
        self.assertEqual([1, 5], s.plusOne([1, 4]))
        self.assertEqual([7, 8, 0], s.plusOne([7, 7, 9]))
        self.assertEqual([1, 0, 0, 0], s.plusOne([9, 9, 9]))

    def test_addBinary(self):
        s = LeetSolution()
        self.assertEqual("100", s.addBinary("11", "1"))
        self.assertEqual("1", s.addBinary("1", "0"))
        self.assertEqual("110", s.addBinary("101", "1"))

    def test_mergeTwoLists(self):
        s = LeetSolution()
        print(s.mergeTwoLists(None, None))
        s.mergeTwoLists(ListNode.makeList([1, 3, 7, 8]), ListNode.makeList([2, 6, 9, 10])).show()
        s.mergeTwoLists(ListNode.makeList([1, 3, 7, 8]), None).show()

    def test_lengthOfLastWord(self):
        s = LeetSolution()
        self.assertEqual(0, s.lengthOfLastWord(""))
        self.assertEqual(5, s.lengthOfLastWord("abcde"))
        self.assertEqual(4, s.lengthOfLastWord("aa dej    djed"))
        self.assertEqual(0, s.lengthOfLastWord("    "))
        self.assertEqual(1, s.lengthOfLastWord("a  "))