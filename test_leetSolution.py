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

    @unittest.skip("succ")
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

    @unittest.skip("succ")
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

    def test_countString(self):
        s = LeetSolution()
        self.assertEqual("11", s.countString("1"))
        self.assertEqual("21", s.countString("11"))
        self.assertEqual("1211", s.countString("21"))
        self.assertEqual("111221", s.countString("1211"))

    def test_countAndSay(self):
        s = LeetSolution()
        self.assertEqual("", s.countAndSay(0))
        self.assertEqual("1", s.countAndSay(1))
        self.assertEqual("11", s.countAndSay(2))
        self.assertEqual("21", s.countAndSay(3))
        self.assertEqual("1211", s.countAndSay(4))
        self.assertEqual("111221", s.countAndSay(5))
        self.assertEqual("312211", s.countAndSay(6))
        self.assertEqual("13112221", s.countAndSay(7))
        self.assertEqual("1113213211", s.countAndSay(8))

    def test_strStr(self):
        s = LeetSolution()
        self.assertEqual(-1, s.strStr("", "abc"))
        self.assertEqual(0, s.strStr("abc", ""))
        self.assertEqual(0, s.strStr("", ""))
        self.assertEqual(2, s.strStr("bcbcdef", "bcd"))
        self.assertEqual(2, s.strStr("ababac", "abac"))
        self.assertEqual(-1, s.strStr("mississippi", "issipi"))

    def test_removeElement(self):
        s = LeetSolution()
        A = []
        self.assertEqual(0, s.removeElement(A, 1))
        self.assertEqual([], A)
        A = [4, 5]
        self.assertEqual(1, s.removeElement(A, 4))
        self.assertEqual([5], A[0:1])
        A = [1]
        self.assertEqual(0, s.removeElement(A, 1))
        self.assertEqual([], A[0:0])
        A = [1, 2, 3, 1]
        self.assertEqual(2, s.removeElement(A, 1))
        self.assertEqual([3, 2], A[0:2])
        A = [1]
        self.assertEqual(1, s.removeElement(A, 2))
        self.assertEqual([1], A)

    def test_removeDuplicates(self):
        s = LeetSolution()
        A = []
        self.assertEqual(0, s.removeDuplicates(A))
        self.assertEqual([], A)
        A = [1, 1, 1, 1]
        self.assertEqual(1, s.removeDuplicates(A))
        self.assertEqual([1], A[0:1])
        A = [1, 1, 2]
        self.assertEqual(2, s.removeDuplicates(A))
        self.assertEqual([1, 2], A[0:2])
        A = [1, 2, 3, 3, 4, 4, 4, 4, 5, 6]
        self.assertEqual(6, s.removeDuplicates(A))
        self.assertEqual([1, 2, 3, 4, 5, 6], A[0:6])
        A = [1, 2, 3, 4, 4, 4, 4]
        self.assertEqual(4, s.removeDuplicates(A))
        self.assertEqual([1, 2, 3, 4], A[0:4])

    def test_isValid(self):
        s = LeetSolution()
        self.assertTrue(s.isValid(""))
        self.assertTrue(s.isValid("()"))
        self.assertTrue(s.isValid("{[()]}"))
        self.assertTrue(s.isValid("()[]{}"))
        self.assertFalse(s.isValid("("))
        self.assertFalse(s.isValid("]"))
        self.assertFalse(s.isValid("(]"))
        self.assertFalse(s.isValid("([)]"))

    @unittest.skip("succ")
    def test_removeNthFromEnd(self):
        s = LeetSolution()
        head = ListNode.makeList([1, 2, 3, 4, 5])
        s.removeNthFromEnd(head, 2).show()
        head = ListNode.makeList([1, 2])
        s.removeNthFromEnd(head, 1).show()
        head = ListNode.makeList([1])
        print(s.removeNthFromEnd(head, 1))
        head = ListNode.makeList([1, 2])
        s.removeNthFromEnd(head, 2).show()
        head = ListNode.makeList([1, 2, 3])
        s.removeNthFromEnd(head, 3).show()

    def test_commonPrefix(self):
        s = LeetSolution()
        self.assertEqual("", s.commonPrefix("", ""))
        self.assertEqual("", s.commonPrefix("", "aaa"))
        self.assertEqual("123", s.commonPrefix("12345", "123567"))

    def test_longestCommonPrefix(self):
        s = LeetSolution()
        self.assertEqual("", s.longestCommonPrefix([]))
        self.assertEqual("1", s.longestCommonPrefix(["1"]))
        self.assertEqual("123", s.longestCommonPrefix(["1234", "123"]))
        self.assertEqual("abc", s.longestCommonPrefix(["abcdef", "abcde", "abcdtg", "abcfff"]))