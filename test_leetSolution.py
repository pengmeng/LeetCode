import unittest
import copy
from unittest import TestCase
from LeetSolution import LeetSolution
from TreeNode import TreeNode
from ListNode import ListNode
from interval import Interval
from random import randint
from random import shuffle

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

    def test_isPalindromeNumber(self):
        s = LeetSolution()
        self.assertTrue(s.isPalindromeNumber(1))
        self.assertTrue(s.isPalindromeNumber(121))
        self.assertTrue(s.isPalindromeNumber(543212345))
        self.assertFalse(s.isPalindromeNumber(123))

    def test_atoi(self):
        s = LeetSolution()
        self.assertEqual(0, s.atoi(""))
        self.assertEqual(1234, s.atoi("1234"))
        self.assertEqual(1234, s.atoi("+1234"))
        self.assertEqual(-1234, s.atoi("-1234"))
        self.assertEqual(1, s.atoi("1"))
        self.assertEqual(0, s.atoi("   a1234"))
        self.assertEqual(1234, s.atoi("   1234  abc"))
        self.assertEqual(-12, s.atoi("  -12a34"))
        self.assertEqual(2147483647, s.atoi("21474836471"))

    def test_reverse(self):
        s = LeetSolution()
        self.assertEqual(0, s.reverse(0))
        self.assertEqual(123, s.reverse(321))
        self.assertEqual(1, s.reverse(100))
        self.assertEqual(-134, s.reverse(-431))

    def test_convert(self):
        s = LeetSolution()
        self.assertEqual("abcds", s.convert("abcds", 1))
        self.assertEqual("PAHNAPLSIIGYIR", s.convert("PAYPALISHIRING", 3))
        self.assertEqual("PINALSIGYAHRPI", s.convert("PAYPALISHIRING", 4))

    def test_findMin(self):
        s = LeetSolution()
        self.assertEqual([], s.findMin([]))
        self.assertEqual(0, s.findMin([0, 1, 2, 4, 5]))
        self.assertEqual(0, s.findMin([4, 5, 6, 7, 0, 1, 2]))

    def test_findMinBinarySearch(self):
        s = LeetSolution()
        self.assertEqual([], s.findMin([]))
        self.assertEqual(1, s.findMinBinarySearch([1, 2]))
        self.assertEqual(1, s.findMinBinarySearch([2, 1]))
        self.assertEqual(0, s.findMinBinarySearch([0, 1, 2, 4, 5]))
        self.assertEqual(0, s.findMinBinarySearch([4, 5, 6, 7, 0, 1, 2]))
        self.assertEqual(1, s.findMinBinarySearch([3, 4, 5, 1, 2]))

    def test_maxProduct(self):
        s = LeetSolution()
        self.assertEqual([], s.maxProduct([]))
        self.assertEqual(6, s.maxProduct([2, 3, -2, 4]))

    def test_evalRPN(self):
        s = LeetSolution()
        self.assertEqual([], s.evalRPN([]))
        self.assertEqual(5, s.evalRPN(["5"]))
        self.assertEqual(25, s.evalRPN(["5", "5", "*"]))
        self.assertEqual(-1, s.evalRPN(["3", "-4", "+"]))
        self.assertEqual(9, s.evalRPN(["2", "1", "+", "3", "*"]))
        self.assertEqual(6, s.evalRPN(["4", "13", "5", "/", "+"]))
        self.assertEqual(22, s.evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]))

    def test_hasCycle(self):
        s = LeetSolution()
        self.assertFalse(s.hasCycle(None))
        A = ListNode.makeList([1])
        self.assertFalse(s.hasCycle(A))
        A.next = A
        self.assertTrue(s.hasCycle(A))
        A = ListNode.makeList([1, 2, 3, 4, 5])
        self.assertFalse(s.hasCycle(A))
        A = ListNode.makeList([1, 2, 3, 4, 5])
        A.tail.next = A.next.next
        self.assertTrue(s.hasCycle(A))

    def test_detectCycle(self):
        s = LeetSolution()
        self.assertEqual(None, s.detectCycle(None))
        A = ListNode.makeList([1])
        self.assertEqual(None, s.detectCycle(A))
        A = ListNode.makeList([1, 2, 3, 4, 5])
        self.assertEqual(None, s.detectCycle(A))
        A = ListNode.makeList([1])
        A.next = A
        self.assertEqual(1, s.detectCycle(A).val)
        A = ListNode.makeList([1, 2, 3, 4, 5])
        A.tail.next = A.next.next
        self.assertEqual(3, s.detectCycle(A).val)

    def test_singleNumber(self):
        s = LeetSolution()
        self.assertEqual([], s.singleNumber([]))
        self.assertEqual(5, s.singleNumber([2, 5, 2]))
        self.assertEqual(1, s.singleNumber([2, 5, 2, 5, 6, 1, 6, 10, 10]))

    def test_singleNumber2(self):
        s = LeetSolution()
        self.assertEqual(1, s.singleNumber2([1]))
        self.assertEqual(1, s.singleNumber2([1, 2, 3, 2, 3, 2, 3, 5, 5, 5]))

    def test_largestNumber(self):
        s = LeetSolution()
        self.assertEqual("9534330", s.largestNumber([3, 30, 34, 5, 9]))
        self.assertEqual("0", s.largestNumber([0, 0]))

    def test_BSTIterator(self):
        s = LeetSolution()
        root = TreeNode.makeTree([3, 1, 4, '#', 2, '#', 5, '#', '#', '#', 6])
        i, r = s.BSTIterator(root), []
        while i.hasNext():
            r.append(i.next())
        self.assertEqual([1, 2, 3, 4, 5, 6], r)

    @unittest.skip("succ")
    def test_wordBreak(self):
        s = LeetSolution()
        in_s = "leetcode"
        in_dict = ["leet", "code"]
        self.assertTrue(s.wordBreak(in_s, in_dict))
        self.assertTrue(s.wordBreak("a", ["a"]))
        self.assertTrue(s.wordBreak("aa", ["a"]))
        self.assertFalse(s.wordBreak("", []))

    @unittest.skip("succ")
    def test_wordBreakii(self):
        s = LeetSolution()
        print(s.wordBreakii("catsanddog", ["cat", "cats", "and", "sand", "dog"]))
        print(s.wordBreakii("aaaaaaaaaab", ["a", "aa", "aaa", "aaaa", "aaaaa"]))
        print(s.wordBreakii("a", []))

    def test_maxProfit(self):
        s = LeetSolution()
        self.assertEqual(6, s.maxProfit([6, 1, 3, 2, 4, 7]))
        self.assertEqual(0, s.maxProfit([5, 4, 3, 2, 1]))
        self.assertEqual(3, s.maxProfit([4, 7, 2, 1]))

    def test_uniquePaths(self):
        s = LeetSolution()
        self.assertEqual(1, s.uniquePaths(1, 1))
        self.assertEqual(1, s.uniquePaths(1, 2))

    def test_uniquePaths2(self):
        s = LeetSolution()
        self.assertEqual(1, s.uniquePaths2(1, 1))
        self.assertEqual(1, s.uniquePaths2(1, 2))

    def test_uniquePathsWithObstacles(self):
        s = LeetSolution()
        self.assertEqual(2, s.uniquePathsWithObstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        self.assertEqual(1, s.uniquePathsWithObstacles([[0, 0]]))
        self.assertEqual(0, s.uniquePathsWithObstacles([[1, 0]]))
        self.assertEqual(0, s.uniquePathsWithObstacles([[0, 0], [1, 1], [0, 0]]))

    def test_uniquePathsWithObstacles2(self):
        s = LeetSolution()
        self.assertEqual(2, s.uniquePathsWithObstacles2([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        self.assertEqual(1, s.uniquePathsWithObstacles2([[0, 0]]))
        self.assertEqual(0, s.uniquePathsWithObstacles2([[1, 0]]))
        self.assertEqual(0, s.uniquePathsWithObstacles2([[0, 0], [1, 1], [0, 0]]))

    def test_minimumTotal(self):
        s = LeetSolution()
        self.assertEqual(1, s.minimumTotal([[1]]))
        self.assertEqual(5, s.minimumTotal([[2], [3, 5]]))
        self.assertEqual(11, s.minimumTotal([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]))
        self.assertEqual(-1, s.minimumTotal([[-1], [2, 3], [1, -1, -3]]))

    def test_numDecodings(self):
        s = LeetSolution()
        self.assertEqual(0, s.numDecodings(""))
        self.assertEqual(0, s.numDecodings("0"))
        self.assertEqual(0, s.numDecodings("00"))
        self.assertEqual(0, s.numDecodings("100"))
        self.assertEqual(1, s.numDecodings("101"))
        self.assertEqual(1, s.numDecodings("10"))
        self.assertEqual(1, s.numDecodings("8"))
        self.assertEqual(2, s.numDecodings("12"))
        self.assertEqual(3, s.numDecodings("1234"))

    def test_numDecodings2(self):
        s = LeetSolution()
        self.assertEqual(0, s.numDecodings2(""))
        self.assertEqual(0, s.numDecodings2("0"))
        self.assertEqual(0, s.numDecodings2("00"))
        self.assertEqual(0, s.numDecodings2("100"))
        self.assertEqual(1, s.numDecodings2("101"))
        self.assertEqual(1, s.numDecodings2("10"))
        self.assertEqual(1, s.numDecodings2("8"))
        self.assertEqual(2, s.numDecodings2("12"))
        self.assertEqual(3, s.numDecodings2("1234"))

    def test_maxSubArray(self):
        s = LeetSolution()
        self.assertEqual(6, s.maxSubArray([4, -1, 2, 1]))

    def test_minPathSum(self):
        s = LeetSolution()
        self.assertEqual(3, s.minPathSum([[1, 2], [1, 1]]))

    def test_longestValidParentheses(self):
        s = LeetSolution()
        self.assertEqual(2, s.longestValidParentheses("(()"))
        self.assertEqual(4, s.longestValidParentheses(")()())"))
        self.assertEqual(2, s.longestValidParentheses("()(()"))
        self.assertEqual(6, s.longestValidParentheses("(()())"))

    def test_canJump(self):
        s = LeetSolution()
        self.assertTrue(s.canJump([2, 3, 1, 1, 4]))
        self.assertFalse(s.canJump([3, 2, 1, 0, 4]))

    def test_candy(self):
        s = LeetSolution()
        self.assertEqual(1, s.candy([1]))
        self.assertEqual(4, s.candy([1, 2, 2]))
        self.assertEqual(6, s.candy([1, 2, 3]))

    def test_jump(self):
        s = LeetSolution()
        self.assertEqual(2, s.jump([2, 3, 1, 1, 4]))
        self.assertEqual(99, s.jump([1] * 100))
        self.assertEqual(499, s.jump([1] * 500))
        self.assertEqual(24999, s.jump([1] * 25000))

    def test_twoSum(self):
        s = LeetSolution()
        self.assertEqual((1, 2), s.twoSum([2, 7, 11, 15], 9))
        self.assertEqual((2, 3), s.twoSum([3, 2, 4], 6))

    @unittest.skip("succ")
    def test_addTwoNumbers(self):
        s = LeetSolution()
        l1 = ListNode.makeList([2, 4, 3])
        l2 = ListNode.makeList([5, 6, 4])
        print(s.addTwoNumbers(l1, l2))

    def test_mergeKLists(self):
        s = LeetSolution()
        values = [sorted([randint(0, 100) for x in range(0, 50)]) for x in range(0, 7)]
        new = copy.deepcopy(values)
        lists = [ListNode.makeList(value) for value in new]
        result = []
        for v in values:
            result.extend(v)
        result.sort()
        self.assertEqual(result, s.mergeKLists(lists).toarray())

    def test_mergeKLists2(self):
        s = LeetSolution()
        values = [sorted([randint(0, 100) for x in range(0, 50)]) for x in range(0, 7)]
        new = copy.deepcopy(values)
        lists = [ListNode.makeList(value) for value in new]
        result = []
        for v in values:
            result.extend(v)
        result.sort()
        self.assertEqual(result, s.mergeKLists2(lists).toarray())

    def test_sortColors(self):
        s = LeetSolution()
        A = [0] * 50 + [1] * 50 + [2] * 50
        shuffle(A)
        s.sortColors(A)
        self.assertEqual([0] * A.count(0) + [1] * A.count(1) + [2] * A.count(2), A)

    def test_maximumGap(self):
        s = LeetSolution()
        A = [15252, 16764, 27963, 7817, 26155, 20757, 3478, 22602, 20404, 6739, 16790, 10588, 16521, 6644, 20880, 15632,
             27078, 25463, 20124, 15728, 30042, 16604, 17223, 4388, 23646, 32683, 23688, 12439, 30630, 3895, 7926,
             22101, 32406, 21540, 31799, 3768, 26679, 21799, 23740]
        self.assertEqual(2901, s.maximumGap(A))

    @unittest.skip("succ")
    def test_mergeinterval(self):
        s = LeetSolution()
        A = [Interval(1, 3), Interval(2, 6), Interval(8, 10), Interval(15, 18)]
        r = s.mergeinterval(A)
        for i in r:
            print(i, end=' ')

    def test_lengthOfLongestSubstring(self):
        s = LeetSolution()
        self.assertEqual(0, s.lengthOfLongestSubstring(''))
        self.assertEqual(3, s.lengthOfLongestSubstring('abcabcbb'))
        self.assertEqual(1, s.lengthOfLongestSubstring('bbbbbbb'))
        self.assertEqual(8, s.lengthOfLongestSubstring('qwnfenpglqdq'))
        self.assertEqual(8, s.lengthOfLongestSubstring('qwnfenpglqdq'))

    def test_insertintervals(self):
        s = LeetSolution()
        r = s.insertintervals(Interval.fromlist([1, 3, 6, 9]), Interval(2, 5))
        self.assertEqual([1, 5, 6, 9], Interval.tolist(r))
        r = s.insertintervals(Interval.fromlist([1, 2, 3, 5, 6, 7, 8, 10, 12, 16]), Interval(4, 9))
        self.assertEqual([1, 2, 3, 10, 12, 16], Interval.tolist(r))

    def test_findRepeatedDnaSequences(self):
        s = LeetSolution()
        # print(s.findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"))

    def test_strstrhash(self):
        s = LeetSolution()
        self.assertEqual(-1, s.strstrhash("", "abc"))
        self.assertEqual(0, s.strstrhash("abc", ""))
        self.assertEqual(0, s.strstrhash("", ""))
        self.assertEqual(2, s.strstrhash("bcbcdef", "bcd"))
        self.assertEqual(2, s.strstrhash("ababac", "abac"))
        self.assertEqual(-1, s.strstrhash("mississippi", "issipi"))

    def test_strstrkmp(self):
        s = LeetSolution()
        self.assertEqual(-1, s.strstrkmp("", "abc"))
        self.assertEqual(0, s.strstrkmp("abc", ""))
        self.assertEqual(0, s.strstrkmp("", ""))
        self.assertEqual(2, s.strstrkmp("bcbcdef", "bcd"))
        self.assertEqual(2, s.strstrkmp("ababac", "abac"))
        self.assertEqual(-1, s.strstrkmp("mississippi", "issipi"))

    def test_divide(self):
        s = LeetSolution()
        self.assertEqual(2, s.divide(4, 2))
        self.assertEqual(2, s.divide(10, 4))
        self.assertEqual(-1, s.divide(1, -1))

    def test_findPeakElement(self):
        s = LeetSolution()
        self.assertEqual(2, s.findPeakElement([1, 2, 3, 1]))
        self.assertEqual(4, s.findPeakElement([1, 2, 3, 4, 5, 3, 2, 1]))
        self.assertEqual(4, s.findPeakElement([1, 2, 3, 4, 5]))
        self.assertEqual(0, s.findPeakElement([5, 4, 3, 2, 1]))