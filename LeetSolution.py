__author__ = 'mengpeng'
from ListNode import ListNode
from TreeNode import TreeNode
from graph import UndirectedGraphNode
import MP_Sort.InsertSort
import functools
import string
import math


class LeetSolution:

    def isPalindrome(self, s):
        i = 0
        j = len(s) - 1
        while i < j:
            if not (s[i].isalpha() or s[i].isdigit()):
                i += 1
            elif not (s[j].isalpha() or s[j].isdigit()):
                j -= 1
            elif not s[i].lower() == s[j].lower():
                return False
            else:
                i += 1
                j -= 1
        return True

    def generate(self, numRows):
        result = []
        for i in range(0, numRows):
            each = []
            for j in range(0, i + 1):
                if j == 0 or j == i:
                    each.append(1)
                else:
                    each.append(result[i-1][j-1] + result[i-1][j])
            result.append(each)
        return result

    def getRow(self, rowIndex):
        result = [1]
        for i in range(0, rowIndex):
            temp = result[0]
            for j in range(1, i + 1):
                ori = result[j]
                result[j] = temp + ori
                temp = ori
            result.append(1)
        return result

    def hasPathSum(self, root, path_sum):
        stack = [(root, root.val)]
        while stack:
            (node, new_sum) = stack.pop()
            if node.right:
                stack.append((node.right, new_sum + node.right.val))
            if node.left:
                stack.append((node.left, new_sum + node.left.val))
            if (not node.right) and (not node.left):
                if new_sum == path_sum:
                    print(new_sum)
                    return True
                else:
                    print(new_sum)
                    continue
        return False

    def minDepth(self, root):
        if not root:
            return 0
        queue = [(root, 1)]
        while queue:
            (node, level) = queue.pop(0)
            if node.left:
                queue.append((node.left, level + 1))
            if node.right:
                queue.append((node.right, level + 1))
            if (not node.right) and (not node.left):
                return level

    def maxDepth(self, root):
        if not root:
            return 0
        stack = [(root, 1)]
        max_level = 1
        while stack:
            (node, level) = stack.pop()
            if node.right:
                stack.append((node.right, level + 1))
            if node.left:
                stack.append((node.left, level + 1))
            if level > max_level:
                max_level = level
            if (not node.right) and (not node.left):
                continue
        return max_level

    def isBalanced(self, root):
        if not root:
            return True
        left_depth = self.depth(root.left)
        right_depth = self.depth(root.right)
        return abs(left_depth - right_depth) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)

    def depth(self, root):
        if not root:
            return 0
        queue = [(root, 1)]
        dep = 0
        while queue:
            (node, dep) = queue.pop(0)
            if node.left:
                queue.append((node.left, dep + 1))
            if node.right:
                queue.append((node.right, dep + 1))
        return dep

    def levelOrder(self, root):
        if not root:
            return []
        queue = [(root, 0)]
        result = []
        cur_dep = -1
        while queue:
            (node, dep) = queue.pop(0)
            if not node:
                continue
            if dep is not cur_dep:
                cur_dep = dep
                result.append([])
            result[dep].append(node.val)
            queue.append((node.left, dep + 1))
            queue.append((node.right, dep + 1))
        return result

    def isSymmetric(self, root):
        if not root:
            return True
        return self.leftRightSym(root.left, root.right)

    def leftRightSym(self, left, right):
        if not left and not right:
            return True
        if not (left and right):
            return False
        if not (left.val == right.val):
            return False
        return self.leftRightSym(left.right, right.left) and self.leftRightSym(left.left, right.right)

    def isSameTree(self, p, q):
        if not (p or q):
            return True
        if not (p and q):
            return False
        if not (p.val == q.val):
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def merge(self, A, m, B, n):
        i = 0
        j = 0
        while j < n:
            if i >= m:
                A.insert(i, B[j])
                j += 1
            elif A[i] > B[j]:
                A.insert(i, B[j])
                j += 1
                m += 1
            i += 1

    def deleteDuplicates(self, head):
        if not head:
            return None
        pred = head
        next_node = head.next
        while next_node:
            if next_node.val != pred.val:
                pred.next = next_node
                pred = next_node
            next_node = next_node.next
        if pred.next:
            pred.next = None
        return head

    def climbStairs(self, n):
        result = [0] * (n + 1)
        result[0] = 1
        result[1] = 1
        for i in range(2, n + 1):
            result[i] = result[i-1] + result[i-2]
        return result[n]

    def plusOne(self, digits):
        index = len(digits) - 1
        last = 1
        while index >= 0:
            temp = digits[index] + last
            if temp > 9:
                last = temp / 10
                digits[index] = temp - 10
            else:
                digits[index] = temp
                return digits
            index -= 1
        if last:
            digits.insert(0, last)
        return digits

    #Pay attention to usage of slice and str functions in this sulotion
    #and multi-value assignment and check clause
    def addBinary(self, a, b):
        reslen = max(len(a), len(b))
        a, b = a.rjust(reslen, '0')[::-1], b.rjust(reslen, '0')[::-1]
        last, res = 0, ''
        for i in range(reslen):
            last, temp = divmod(int(a[i]) + int(b[i]) + last, 2)
            res += str(temp)
        if last:
            res += '1'
        return res[::-1]

    def mergeTwoLists(self, l1, l2):
        headone = l1
        headtwo = l2
        reshead = ListNode(0)
        tail = reshead
        while headone or headtwo:
            if not headone:
                tail.next = headtwo
                return reshead.next
            if not headtwo:
                tail.next = headone
                return reshead.next
            if headone.val < headtwo.val:
                tail.next = headone
                headone = headone.next
            else:
                tail.next = headtwo
                headtwo = headtwo.next
            tail = tail.next
        return reshead.next

    def lengthOfLastWord(self, s):
        if not s:
            return 0
        s = s.rstrip()
        lastspace = s.rfind(' ')
        return len(s) - lastspace - 1

    def countAndSay(self, n):
        last_result = ""
        for i in range(1, n + 1):
            last_result = self.countString(last_result)
        return last_result

    def countString(self, s):
        if not s:
            return "1"
        last_value = int(s[0])
        last_count = 1
        result = ""
        for i in range(1, len(s)):
            new_value = int(s[i])
            if new_value == last_value:
                last_count += 1
            else:
                result += str(last_count) + str(last_value)
                last_count = 1
                last_value = new_value
        result += str(last_count) + str(last_value)
        return result

    def strStrMy(self, haystack, needle):
        index_hk, len_hk = 0, len(haystack)
        index_ne, len_nd = 0, len(needle)
        if len_nd == 0:
            return 0
        while index_hk < len_hk and index_ne < len_nd:
            if haystack[index_hk] == needle[index_ne]:
                last_pos = index_hk
                while index_hk < len_hk:
                    if haystack[index_hk] == needle[index_ne]:
                        index_hk += 1
                        index_ne += 1
                    else:
                        index_hk = last_pos + 1
                        index_ne = 0
                        break
                    if index_ne == len_nd:
                        return last_pos
            else:
                index_hk += 1
        return -1

    def strStr(self, haystack, needle):
        index_hk, len_hk = 0, len(haystack)
        index_ne, len_nd = 0, len(needle)
        while True:
            index_ne = 0
            while True:
                if index_ne == len_nd:
                    return index_hk
                if index_hk + index_ne == len_hk:
                    return -1
                if haystack[index_hk + index_ne] != needle[index_ne]:
                    break
                index_ne += 1
            index_hk += 1

    def removeElement(self, A, elem):
        length = len(A)
        index = 0
        while index < length:
            if A[index] == elem:
                A[index] = A[length - 1]
                length -= 1
            else:
                index += 1
        return index

    def removeDuplicates(self, A):
        i, j = 0, 0
        length = len(A)
        if length == 0:
            return 0
        while j < length:
            if (j == length - 1 or A[j+1] != A[j]) and (A[j] != A[i]):
                if j - i > 1:
                    A[i+1] = A[j]
                i += 1
            j += 1
        return i + 1

    #Valid Parentheses
    #Python doesn't have switch-case like statement
    def isValid(self, s):
        index = 0
        length = len(s)
        stack = []
        while index < length:
            if s[index] == '(' or s[index] == '[' or s[index] == '{':
                stack.append(s[index])
            elif s[index] == ')':
                if not stack or stack.pop() != '(':
                    return False
            elif s[index] == ']':
                if not stack or stack.pop() != '[':
                    return False
            elif s[index] == '}':
                if not stack or stack.pop() != '{':
                    return False
            index += 1
        if stack and length != 0:
            return False
        else:
            return True

    def removeNthFromEnd(self, head, n):
        pred = head
        tail = head
        while n > 0:
            tail = tail.next
            n -= 1
        while tail and tail.next:
            pred = pred.next
            tail = tail.next
        if not tail:
            head = head.next
        else:
            pred.next = pred.next.next
        return head

    #Trie can be used to solve this problem but I'm not sure about efficiency yet
    #Will try later
    def longestCommonPrefix(self, strs):
        index, length = 1, len(strs)
        if length == 0:
            return ""
        prefix = strs[0]
        while index < length:
            prefix = self.commonPrefix(prefix, strs[index])
            index += 1
        return prefix

    def commonPrefix(self, string1, string2):
        index = 0
        length = min(len(string1), len(string2))
        while index < length:
            if string1[index] != string2[index]:
                break
            index += 1
        return string1[0:index]

    #Palindrome Number
    #Attention here! If reversed number is overflow, how to sovle?
    def isPalindromeNumber(self, x):
        if x < 0:
            return False
        temp, rev = x, 0
        while temp > 0:
            rev = rev * 10 + temp % 10
            temp //= 10
        return rev == x

    def atoi(self, str):
        index, length = 0, len(str)
        result, factor = 0, 0
        MAX, MIN = 2147483647, -2147483648
        while index < length and str[index] == ' ':
            index += 1
        if index == length:
            return result
        elif str[index] == '+':
            factor = 1
            index += 1
        elif str[index] == '-':
            factor = -1
            index += 1
        elif str[index].isdigit():
            factor = 1
        else:
            return result
        while index < length and str[index].isdigit():
            result = result * 10 + int(str[index])
            index += 1
        result *= factor
        if result > MAX:
            return MAX
        elif result < MIN:
            return MIN
        else:
            return result

    #Reverse Integer
    def reverse(self, x):
        MAX, MIN = 2147483647, -2147483648
        result = 0
        factor = 1 if x > 0 else -1
        x = abs(x)
        while abs(x) != 0:
            result = result * 10 + x % 10
            x //= 10
        result *= factor
        if result > MAX or result < MIN:
            return 0
        else:
            return result

    #ZigZag Conversion
    #More Attention!
    #Use varible step to control loop varible row to move between two edges
    #Find a new way to intialize multiple dimension array in Python
    def convert(self, s, nRows):
        if nRows <= 1:
            return s
        result = []
        for i in range(nRows):
            result.append("")
        row, step = 0, 1
        for i in range(len(s)):
            result[row] += s[i]
            if row == 0:
                step = 1
            elif row == nRows - 1:
                step = -1
            row += step
        s = ""
        for i in range(nRows):
            s += result[i]
        return s

    #Find Minimum in Rotated Sorted Array
    def findMin(self, num):
        for i in range(1, len(num)):
            if num[i - 1] > num[i]:
                return num[i]
        return num and num[0]

    def findMinBinarySearch(self, num):
        length = len(num)
        if not num or num[0] < num[length - 1]:
            return num and num[0]
        left, right = 0, length - 1
        while left < right:
            mid = (left + right) >> 1
            if num[mid] > num[right]:
                left = mid + 1
            else:
                right = mid
        return num[left]

    #Consideration...
    def maxProduct(self, A):
        front, back, length = 1, 1, len(A)
        result = A and A[0]
        for i in range(length):
            front *= A[i]
            back *= A[length-i-1]
            result = max(result, max(front, back))
            front = 1 if front == 0 else front
            back = 1 if back == 0 else back
        return result

    #Evaluate Reverse Polish Notation
    #Pay attention to difference of / and // and other similar calculation in Python
    def evalRPN(self, tokens):
        index, length = 0, len(tokens)
        stack = []
        while index < length:
            token = tokens[index]
            if token == '+':
                stack.append(stack.pop() + stack.pop())
            elif token == '-':
                second = stack.pop()
                stack.append(stack.pop() - second)
            elif token == '*':
                stack.append(stack.pop() * stack.pop())
            elif token == '/':
                second = stack.pop()
                stack.append(int(float(stack.pop()) / second))
            else:
                stack.append(int(token))
            index += 1
        if stack:
            return stack.pop()
        else:
            return stack

    def sortList(self, head):
        pass

    #Following two questions make world a marvellous one
    #Linked List Cycle
    def hasCycle(self, head):
        if not head:
            return False
        slower, faster = head, head
        while slower and faster and faster.next:
            slower = slower.next
            faster = faster.next.next
            if slower is faster:
                return True
        return False

    #Linked List Cycle II
    def detectCycle(self, head):
        if not head:
            return None
        slower, faster = head, head
        while slower and faster and faster.next:
            slower = slower.next
            faster = faster.next.next
            if slower is faster:
                faster = head
                while slower and faster:
                    if slower is faster:
                        return slower
                    slower = slower.next
                    faster = faster.next
        return None

    #In python3 reduce() function has been move to functools
    def singleNumber(self, A):
        import functools
        return A and functools.reduce(lambda x, y: x ^ y, A)

    def insertionSortList(self, head):
        return MP_Sort.InsertSort.insertsort_list(head)

    #Such a perfect algorithm
    #I take it from dicussion
    def singleNumber2(self, A):
        ones, twos = 0, 0
        for item in A:
            ones = (ones ^ item) & ~twos
            twos = (twos ^ item) & ~ones
        return ones

    def reorderList(self, head):
        pass

    def largestNumber(self, num):
        num_str = [str(x) for x in num]
        num_str.sort(key=functools.cmp_to_key(lambda x, y: int(x+y) - int(y+x)), reverse=True)
        result = ''.join(num_str)
        return result.lstrip('0') or '0'

    #Binary Search Tree Iterator
    class BSTIterator:
        # @param root, a binary search tree's root node
        def __init__(self, root):
            self.queue = []
            stack = [root]
            while stack:
                node = stack.pop()
                while node:
                    stack.append(node)
                    node = node.left
                if stack:
                    node = stack.pop()
                    self.queue.append(node.val)
                    stack.append(node.right)

        # @return a boolean, whether we have a next smallest number
        def hasNext(self):
            return self.queue

        # @return an integer, the next smallest number
        def next(self):
            return self.queue.pop(0)

    #Word Break
    #forward DP
    def wordBreak(self, s, dict):
        if not s or not dict or s in dict:
            return s and s in dict
        length = len(s)
        subseq, subseq[0] = [False for x in range(0, length + 1)], True
        for i in range(1, length + 1):
            for j in range(0, i):
                if subseq[j]:
                    if s[j:i] in dict:
                        subseq[i] = True
                        break
        return subseq[length]

    #Word Break II
    #backward DP
    def wordBreakii(self, s, dict):
        if not s or not dict:
            return []
        length = len(s)
        subseq, subseq[length] = [False for x in range(0, length + 1)], True
        subset, subset[length] = [[] for x in range(0, length + 1)], [""]
        for i in range(length - 1, -1, -1):
            for j in range(length, i, -1):
                if subseq[j]:
                    if s[i:j] in dict:
                        subseq[i] = True
                        if subset[j]:
                            for sufix in subset[j]:
                                if sufix is "":
                                    subset[i].append(s[i:j] + sufix)
                                else:
                                    subset[i].append(s[i:j] + " " + sufix)
        return subset[0]

    #Best Time to Buy and Sell Stock
    def maxProfit(self, prices):
        if not prices:
            return 0
        curmax, lowest, highest = 0, prices[0], prices[0]
        for price in prices[1:]:
            if price > highest:
                highest = price
            elif price < lowest:
                curmax = max(curmax, highest - lowest)
                highest, lowest = price, price
        return max(curmax, highest - lowest)

    #Unique Paths
    def uniquePaths(self, m, n):
        solution = [[1 if x == 0 or y == 0 else 0 for x in range(n)] for y in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                solution[i][j] = solution[i-1][j] + solution[i][j-1]
        return solution[m - 1][n - 1]

    #Unique Paths w/ 1D array DP
    def uniquePaths2(self, m, n):
        solution = [1 for x in range(n)]
        for i in range(1, m):
            for j in range(1, n):
                solution[j] += solution[j - 1]
        return solution[n - 1]

    #Unique Paths II
    def uniquePathsWithObstacles(self, obstacleGrid):
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        solution = [[0 for x in range(n+1)] for y in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if obstacleGrid[i-1][j-1] == 1:
                    solution[i][j] = 0
                elif i == 1 and j == 1:
                    solution[i][j] = 1 - obstacleGrid[i-1][j-1]
                else:
                    solution[i][j] = solution[i-1][j] + solution[i][j-1]
        return solution[m][n]

    #Unique Paths II w/ 1D array DP
    def uniquePathsWithObstacles2(self, obstacleGrid):
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        solution = [0 for x in range(n)]
        solution[0] = 1
        for i in range(0, m):
            for j in range(0, n):
                if obstacleGrid[i][j] == 1:
                    solution[j] = 0
                elif j > 0:
                    solution[j] += solution[j - 1]
        return solution[n - 1]

    #Triangle w/ bottom-up DP
    def minimumTotal(self, triangle):
        level = len(triangle)
        result = triangle[level - 1]
        for i in range(level - 2, -1, -1):
            for j in range(i + 1):
                result[j] = triangle[i][j] + min(result[j], result[j + 1])
        return result[0]

    #Decode Ways
    def numDecodings(self, s):
        length = len(s)
        if length == 0:
            return 0
        result = [0 for x in range(length + 1)]
        result[0] = 1
        result[1] = 0 if s[0] is '0' else 1
        for i in range(1, length):
            if s[i] is not '0':
                result[i + 1] += result[i]
            if int(s[i - 1:i + 1]) <= 26 and s[i - 1] is not '0':
                result[i + 1] += result[i - 1]
        return result[length]

    #Decode Ways w/ O(1) space
    def numDecodings2(self, s):
        length = len(s)
        if length == 0:
            return 0
        fisrt = 1
        second = 0 if s[0] is '0' else 1
        for i in range(1, length):
            x, y = 0, 0
            if s[i] is not '0':
                x = second
            if int(s[i - 1:i + 1]) <= 26 and s[i - 1] is not '0':
                y = fisrt
            fisrt = second
            second = x + y
        return second

    #Maximum Subarray
    def maxSubArray(self, A):
        finalmax, curmax = A[0], A[0]
        for i in range(1, len(A)):
            curmax = max(A[i], curmax + A[i])
            if curmax > finalmax:
                finalmax = curmax
        return finalmax

    #Minimum Path Sum
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])
        solution = [0 for x in range(n)]
        for i in range(m):
            for j in range(n):
                if i is 0 and j is 0:
                    solution[j] = grid[i][j]
                elif j is 0:
                    solution[j] += grid[i][j]
                elif i is 0:
                    solution[j] = solution[j - 1] + grid[i][j]
                else:
                    solution[j] = min(solution[j - 1], solution[j]) + grid[i][j]
        return solution[n - 1]

    #Longest Valid Parentheses
    def longestValidParentheses(self, s):
        length = len(s)
        longest, curmax = [0 for x in range(length)], 0
        for i in range(1, length):
            if s[i] is ')':
                if s[i - 1] is '(':
                    longest[i] = longest[i - 2] + 2 if i - 2 >= 0 else 2
                    curmax = max(curmax, longest[i])
                else:
                    prematch = i - longest[i - 1] - 1
                    if prematch >= 0 and s[prematch] is '(':
                        longest[i] = longest[i - 1] + 2 + (longest[prematch - 1] if prematch - 1 >= 0 else 0)
                        curmax = max(curmax, longest[i])
        return curmax

    #Unique Binary Search Trees
    def numTrees(self, n):
        result = [0 for x in range(n + 1)]
        result[0], result[1] = 1, 1
        for i in range(2, n + 1):
            for j in range(0, i):
                result[i] += result[j] * result[i - j - 1]
        return result[n]

    #Unique Binary Search Trees w/ CATALAN NUMBER
    def numTrees2(self, n):
        result = 1
        for i in range(1, n + 1):
            result = result * 2 * (2 * i - 1) / (i + 1)
        return result

    #Gas Station
    def canCompleteCircuit(self, gas, cost):
        start, remain, need = 0, 0, 0
        for i in range(len(gas)):
            remain = remain + gas[i] - cost[i]
            if remain < 0:
                need += remain
                start, remain = i + 1, 0
        return -1 if remain + need < 0 else start

    #Best Time to Buy and Sell Stock II
    def maxProfit2(self, prices):
        total = 0
        for i in range(len(prices) - 1):
            if prices[i + 1] > prices[i]:
                total += prices[i + 1] - prices[i]
        return total

    #Jump Game
    def canJump(self, A):
        reach, length, i = 0, len(A), 0
        while i < length and i <= reach:
            reach = reach if i + A[i] <= reach else i + A[i]
            i += 1
        return i == length

    #Candy
    def candy(self, ratings):
        length, count = len(ratings), [1]
        if length < 2:
            return length
        for i in range(1, length):
            if ratings[i] > ratings[i - 1]:
                count.append(count[i - 1] + 1)
            else:
                count.append(1)
        for i in range(length - 1, 0, -1):
            if ratings[i - 1] > ratings[i]:
                count[i - 1] = max(count[i] + 1, count[i - 1])
        return sum(count)

    #Jump Game
    def jump(self, A):
        length, i = len(A), 0
        maxreach, reach, count = 0, 0, 0
        while i < length - 1 and i <= reach:
            maxreach = max(maxreach, i + A[i])
            if i == reach:
                #cool stuff here in python integer -5 ~ 257 defined in a array and will be referred to the index
                #integer out of this range will result in different id() and cannot be compared by is
                reach = maxreach
                count += 1
            i += 1
        return count if reach >= length - 1 else -1

    #Two Sum
    def twoSum(self, num, target):
        table = {num[0]: 0}
        for i in range(1, len(num)):
            first = table.get(target - num[i])
            if first is not None:
                return first + 1, i + 1
            else:
                table[num[i]] = i

    #Add Two Numbers
    def addTwoNumbers(self, l1, l2):
        result, carry = ListNode(0), 0
        pt = result
        while l1 or l2 or carry:
            pt.next = ListNode(carry)
            pt = pt.next
            if l1:
                pt.val += l1.val
                l1 = l1.next
            if l2:
                pt.val += l2.val
                l2 = l2.next
            carry = pt.val // 10
            pt.val %= 10
        return result.next

    #Anagrams
    def anagrams(self, strs):
        import collections
        d = collections.defaultdict(list)
        for s in strs:
            d[tuple(sorted(s))].append(s)
        return [s for group in d.values() if len(group) > 1 for s in group]

    #Merge k Sorted Lists
    #Divide-and-Conquer
    def mergeKLists(self, lists):
        length = len(lists)
        if length == 0:
            return None
        if length == 1:
            return lists[0]
        pivot = length // 2
        left = self.mergeKLists(lists[0: pivot])
        right = self.mergeKLists(lists[pivot:])
        tail = head = ListNode(0)
        while left or right:
            if not left or (right and right.val < left.val):
                tail.next = ListNode(right.val)
                right = right.next
            else:
                tail.next = ListNode(left.val)
                left = left.next
            tail = tail.next
        return head.next

    #Merge k Sorted Lists
    #heap
    def mergeKLists2(self, lists):
        import heapq
        heap = [node for node in lists if node]
        tail = head = ListNode(0)
        heapq.heapify(heap)
        while heap:
            tail.next = heapq.heappop(heap)
            tail = tail.next
            if tail.next:
                heapq.heappush(heap, tail.next)
        return head.next

    #Sort Colors
    def sortColors(self, A):
        low, high, i = 0, len(A) - 1, 0
        while i <= high:
            if A[i] == 0:
                A[i] = A[low]
                A[low] = 0
                i += 1
                low += 1
            elif A[i] == 2:
                A[i] = A[high]
                A[high] = 2
                high -= 1
            else:
                i += 1

    #Maximum Gap
    def maximumGap(self, num):
        length = len(num)
        if length < 2:
            return 0
        minv, maxv = min(num), max(num)
        gap = (maxv - minv - 1) / (length - 1) + 1
        backetnum = int((maxv - minv) / gap) + 1
        minbacket, maxbacket = [maxv+1 for x in range(backetnum)], [minv-1 for x in range(backetnum)]
        for i in range(length):
            if num[i] != minv and num[i] != maxv:
                backeti = int((num[i] - minv) / gap)
                minbacket[backeti] = min(minbacket[backeti], num[i])
                maxbacket[backeti] = max(maxbacket[backeti], num[i])
        maxgap, prev = -1, minv
        for i in range(backetnum):
            if minbacket[i] != maxv+1 and maxbacket[i] != minv-1:
                maxgap = max(maxgap, minbacket[i] - prev)
                prev = maxbacket[i]
        return max(maxgap, maxv - prev)

    #Merge Intervals
    def mergeinterval(self, intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x.start)
        result = [intervals[0]]
        for each in intervals[1:]:
            if each.start <= result[-1].end:
                result[-1].end = max(each.end, result[-1].end)
            else:
                result.append(each)
        return result

    #Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s):
        start, maxlen = 0, 0
        used = {}
        for i in range(len(s)):
            c = s[i]
            if c in used and used[c] >= start:
                start = used[c] + 1
                used[c] = i
            else:
                used[c] = i
                maxlen = max(maxlen, i - start + 1)
        return maxlen

    #Insert Interval
    def insertintervals(self, intervals, newInterval):
        new, result = newInterval, []
        for i in range(len(intervals)):
            cur = intervals[i]
            if cur.end < new.start:
                result.append(cur)
            elif cur.start > new.end:
                result.append(new)
                return result + intervals[i:]
            else:
                new.start = min(cur.start, new.start)
                new.end = max(cur.end, new.end)
        result.append(new)
        return result

    #Repeated DNA Sequences
    def findRepeatedDnaSequences(self, s):
        subs, result, length = {}, [], len(s)
        if length <= 10:
            return []
        for i in range(length - 9):
            sub = s[i:i + 10]
            subs[sub] = subs.get(sub, 0) + 1
        return [k for k in iter(subs) if subs[k] > 1]

    #strStr() using hash
    #Known as Rabin-Karp algorithm
    def strstrhash(self, haystack, needle):
        nhash, nlen = hash(needle), len(needle)
        hlen = len(haystack)
        if nlen == 0:
            return 0
        for i in range(hlen - nlen + 1):
            if hash(haystack[i:i + nlen]) == nhash:
                return i
        return -1

    #strStr() using KMP algorithm
    #I have no idea why this is slower than RK on leetcode
    def strstrkmp(self, haystack, needle):
        hlen, nlen = len(haystack), len(needle)
        kmpn = [0] * nlen
        j = 0
        for i in range(1, nlen):
            while j > 0 and needle[j] is not needle[i]:
                j = kmpn[j - 1]
            if needle[j] is needle[i]:
                j += 1
            kmpn[i] = j
        if nlen == 0:
            return 0
        j = 0
        for i in range(hlen):
            while j > 0 and needle[j] is not haystack[i]:
                j = kmpn[j - 1]
            if needle[j] is haystack[i]:
                j += 1
            if j == nlen:
                return i - nlen + 1
        return -1

    #Max Points on a Line
    def maxPoints(self, points):
        length = len(points)
        if length <= 2:
            return length
        maxpoint = 2
        for i in range(length - 2):
            maxcur, same, table = 1, 0, {}
            for j in range(i+1, length):
                dx = float(points[j].x) - float(points[i].x)
                dy = float(points[j].y) - float(points[i].y)
                if abs(dx) < 0.01 and abs(dy) < 0.01:
                    same += 1
                    continue
                ratio = abs(dy) / (abs(dx) + abs(dy))
                ratio = str(ratio) if dx * dy > 0 else str(-ratio)
                if ratio in table:
                    table[ratio] += 1
                else:
                    table[ratio] = 2
                if maxcur < table[ratio]:
                    maxcur = table[ratio]
            maxcur += same
            maxpoint = max(maxcur, maxpoint)
        return maxpoint

    table = {}

    #Clone Graph w/ DFS recursion
    def cloneGraph(self, node):
        if not node:
            return None
        start = UndirectedGraphNode(node.label)
        LeetSolution.table[node] = start
        for each in node.neighbors:
            if each in LeetSolution.table:
                start.neighbors.append(LeetSolution.table[each])
            else:
                neighbor = self.cloneGraph(each)
                start.neighbors.append(neighbor)
        return start

    #Clone Graph w/ DFS not recursion
    def cloneGraph2(self, node):
        if not node:
            return None
        visited = {}
        stack = [node]
        head = UndirectedGraphNode(node.label)
        visited[node] = head
        while stack:
            cur = stack.pop()
            for each in cur.neighbors:
                if not each in visited:
                    stack.append(each)
                    newnode = UndirectedGraphNode(each.label)
                    visited[each] = newnode
                    visited[cur].neighbors.append(newnode)
                else:
                    visited[cur].neighbors.append(visited[each])
        return head

    #Divide Two Integers
    def divide(self, dividend, divisor):
        positive = (dividend > 0) is (divisor > 0)
        MIN_INT, MAX_INT = -2147483648, 2147483647
        if divisor == 0:
            return MAX_INT
        dividend, divisor = abs(dividend), abs(divisor)
        result = 0
        while dividend >= divisor:
            temp, factor = divisor, 1
            while dividend >= temp:
                dividend -= temp
                result += factor
                factor <<= 1
                temp <<= 1
        if not positive:
            result = -result
        return min(max(MIN_INT, result), MAX_INT)

    #Find Peak Element
    def findPeakElement(self, num):
        left, right, length = 0, len(num) - 1, len(num)
        while left <= right:
            mid = (left + right) >> 1
            if mid + 1 < length and num[mid] < num[mid + 1]:
                left = mid + 1
            elif mid - 1 >= 0 and num[mid] < num[mid - 1]:
                right = mid - 1
            else:
                return mid
        return left

    #Search Insert Position
    def searchInsert(self, A, target):
        length = len(A)
        left, right = 0, length - 1
        while left <= right:
            mid = (left + right) >> 1
            if A[mid] < target:
                left = mid + 1
            elif A[mid] > target:
                right = mid - 1
            else:
                return mid
        return left

    #Binary Tree Zigzag Level Order Traversal
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        queue = [(root, 0)]
        result = []
        cur_dep = -1
        while queue:
            (node, dep) = queue.pop(0)
            if not node:
                continue
            if dep is not cur_dep:
                cur_dep = dep
                result.append([])
            result[dep].append(node.val)
            queue.append((node.left, dep + 1))
            queue.append((node.right, dep + 1))
        for i in range(len(result)):
            if i % 2:
                result[i].reverse()
        return result

    #Word Ladder
    def ladderLength(self, start, end, dict):
        distance = {start: 1}
        queue = [start]
        while queue:
            word = queue.pop(0)
            for i in range(len(word)):
                for l in string.ascii_lowercase:
                    newword = word[:i] + l + word[i+1:]
                    if newword == end:
                        return distance[word] + 1
                    if newword in dict and newword not in distance:
                        queue.append(newword)
                        distance[newword] = distance[word] + 1
        return 0

    #Majority Element
    #Moore Voting
    def majorityElement(self, num):
        candidate, count = num[0], 1
        for each in iter(num[1:]):
            if count is 0:
                candidate, count = each, 1
            elif candidate == each:
                count += 1
            else:
                count -= 1
        return candidate

    #Sqrt(x)
    #Newton's Iterative
    def sqrt(self, x):
        ans, delta = float(x), 0.0001
        while abs(ans ** 2 - x) > delta:
            ans = (ans + x / ans) / 2
        return ans

    #Search for a Range
    #Tricky solution
    def searchRange(self, A, target):
        start = self.bsearch(A, target - 0.5)
        if A[start] != target:
            return [-1, -1]
        A.append(0)
        end = self.bsearch(A, target + 0.5) - 1
        return [start, end]

    def bsearch(self, A, target):
        left, right = 0, len(A) - 1
        while left < right:
            mid = (left + right) >> 1
            if A[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    #Search a 2D Matrix
    def searchMatrix(self, matrix, target):
        left, right = 0, len(matrix) - 1
        while left <= right:
            mid = (left + right) >> 1
            if matrix[mid][0] < target:
                left = mid + 1
            elif matrix[mid][0] > target:
                right = mid - 1
            else:
                return True
        if left - 1 < 0:
            return False
        line = matrix[left - 1]
        left, right = 0, len(line) - 1
        while left <= right:
            mid = (left + right) >> 1
            if line[mid] < target:
                left = mid + 1
            elif line[mid] > target:
                right = mid - 1
            else:
                return True
        return False

    #Find Minimum in Rotated Sorted Array II
    def findMinii(self, num):
        left, right = 0, len(num) - 1
        while left < right:
            mid = (left + right) >> 1
            if num[mid] > num[right]:
                left = mid + 1
            elif num[mid] < num[right]:
                right = mid
            else:
                right -= 1
        return num[left]

    #Search in Rotated Sorted Array
    def search(self, A, target):
        left, right = 0, len(A) - 1
        while left <= right:
            mid = (left + right) >> 1
            if A[mid] == target:
                return mid
            if A[left] <= A[mid]:
                if A[left] <= target < A[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if A[mid] < target <= A[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    #Search in Rotated Sorted Array II
    def searchii(self, A, target):
        left, right = 0, len(A) - 1
        while left <= right:
            mid = (left + right) >> 1
            if A[mid] == target:
                return True
            if A[left] < A[mid]:
                if A[left] <= target < A[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            elif A[left] > A[mid]:
                if A[mid] < target <= A[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                left += 1
        return False

    #MinStack
    class MinStack:
        def __init__(self):
            self.stack = []

        # @param x, an integer
        # @return an integer
        def push(self, x):
            curmin = self.getMin()
            if curmin is None or x < curmin:
                curmin = x
            self.stack.append((x, curmin))

        # @return nothing
        def pop(self):
            self.stack.pop()

        # @return an integer
        def top(self):
            return self.stack[-1][0] if self.stack else None

        # @return an integer
        def getMin(self):
            return self.stack[-1][1] if self.stack else None

    #LRU Cache
    class LRUCache:
        def __init__(self, capacity):
            import collections
            self.capacity = capacity
            self.cache = collections.OrderedDict()

        # @return an integer
        def get(self, key):
            value = -1
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
            return value

        # @param key, an integer
        # @param value, an integer
        # @return nothing
        def set(self, key, value):
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) == self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value

    #Number of 1 Bits
    def hammingWeight(self, n):
        #return bin(n).count('1')
        count = 0
        while n:
            n &= n-1
            count += 1
        return count

    #Reverse Bits
    def reverseBits(self, n):
        #return int(bin(n)[:1:-1].ljust(32,'0'), 2)
        r = 0
        for i in range(32):
            r += (n >> i & 1) << (31-i)
        return r

    #Reverse Integer
    def reverseinteger(self, x):
        r = int(str(abs(x))[::-1])*(1 if x >= 0 else -1)
        return r if -2147483648 < r < 2147483647 else 0

    #Bitwise AND of Numbers Range
    def rangeBitwiseAnd(self, m, n):
        pass

    #Binary Tree Right Side View
    def rightSideView(self, root):
        if not root:
            return []
        result, queue = [], [(root, 0)]
        while queue:
            node, level = queue.pop(0)
            if node:
                if level >= len(result):
                    result.append(node.val)
                queue.append((node.right, level+1))
                queue.append((node.left, level+1))
        return result

    #Number of Islands
    def numIslands(self, grid):
        if len(grid) == 0:
            return 0
        n, m, count = len(grid), len(grid[0]), 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    self._isisland(grid, i, j, n, m)
                    count += 1
        return count

    def _isisland(self, grid, i, j, n, m):
        if i < 0 or j < 0 or i >= n or j >= m or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        self._isisland(grid, i+1, j, n, m)
        self._isisland(grid, i, j+1, n, m)
        self._isisland(grid, i-1, j, n, m)
        self._isisland(grid, i, j-1, n, m)

    #Surrounded Regions
    def solve(self, board):
        if len(board) == 0:
            return
        board[:] = [list(x) for x in board]
        n, m = len(board), len(board[0])
        for i in [0, n - 1]:
            for j in range(m):
                if board[i][j] == 'O':
                    self.bfssolve(board, i, j, n, m)
        for i in range(n):
            for j in [0, m - 1]:
                if board[i][j] == 'O':
                    self.bfssolve(board, i, j, n, m)
        for i in range(n):
            for j in range(m):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'V':
                    board[i][j] = 'O'
        board[:] = [''.join(x) for x in board]

    #dfs will cause max recursion depth exceeded
    def dfssolve(self, board, i, j, n, m):
        if i < 0 or j < 0 or i >= n or j >= m:
            return False
        elif board[i][j] == 'X':
            return True
        board[i][j] = 'X'
        flag = self.dfssolve(board, i+1, j, n, m) \
               and self.dfssolve(board, i, j+1, n, m) \
               and self.dfssolve(board, i-1, j, n, m) \
               and self.dfssolve(board, i, j-1, n, m)
        board[i][j] = 'O'
        return flag

    def bfssolve(self, board, i, j, n, m):
        queue, board[i][j] = [], 'V'
        queue.append((i, j))
        while queue:
            i, j = queue.pop(0)
            if i > 0 and board[i-1][j] == 'O':
                board[i-1][j] = 'V'
                queue.append((i-1, j))
            if i < n - 1 and board[i+1][j] == 'O':
                board[i+1][j] = 'V'
                queue.append((i+1, j))
            if j > 0 and board[i][j-1] == 'O':
                board[i][j-1] = 'V'
                queue.append((i, j-1))
            if j < m - 1 and board[i][j+1] == 'O':
                board[i][j+1] = 'V'
                queue.append((i, j+1))

    #Simplify Path
    def simplifyPath(self, path):
        parts, stack = path.split('/'), []
        for c in parts:
            if c == '' or c == '.':
                continue
            elif c == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(c)
        return '/' + '/'.join(stack)

    #Trapping Rain Water
    def trap(self, height):
        water, left, right = 0, 0, 0
        i, j = 0, len(height) - 1
        while i <= j:
            left, right = max(height[i], left), max(height[j], right)
            while i <= j and height[i] <= left <= right:
                water += left - height[i]
                i += 1
            while i <= j and height[j] <= right <= left:
                water += right - height[j]
                j -= 1
        return water

    #Maximal Rectangle
    def maximalRectangle(self, matrix):
        if not matrix:
            return 0
        n, m = len(matrix), len(matrix[0])
        left, right, height = [0]*m, [m]*m, [0]*m
        maxarea = 0
        for i in range(n):
            curleft, curright = 0, m
            for j in range(m):
                if matrix[i][j] == '1':
                    left[j] = max(left[j], curleft)
                    height[j] += 1
                else:
                    curleft = j + 1
                    left[j], height[j] = 0, 0
                if matrix[i][m-j-1] == '1':
                    right[m-j-1] = min(right[m-j-1], curright)
                else:
                    right[m-j-1] = m
                    curright = m - j - 1
            for j in range(m):
                maxarea = max(maxarea, (right[j] - left[j]) * height[j])
        return maxarea

    #Permutations
    def permute(self, nums):
        return [nums] if len(nums) <= 1 else [[x] + y for x in nums for y in self.permute(nums[:nums.index(x)]+nums[nums.index(x)+1:])]

    #Permutations II
    def permuteii(self, nums):
        result = {()}
        for n in nums:
            result = {r[:i]+(n,)+r[i:] for r in result for i in range(len(r)+1)}
        return [list(x) for x in result]

    #Permutation Sequence
    def getPermutation(self, n, k):
        import math
        nums, result = [i + 1 for i in range(n)], ''
        while n > 0:
            totalCount = math.factorial(n - 1)
            index = (k - 1) // totalCount
            result += str(nums.pop(index))
            n -= 1
            k %= totalCount
        return result

    #Gray Code
    def grayCode(self, n):
        return [(i >> 1) ^ i for i in range(2**n)]

    #Word Search
    def exist(self, board, word):
        if not word:
            return True
        if len(board) == 0:
            return False
        board = [list(x) for x in board]
        n, m = len(board), len(board[0])
        for i in range(n):
            for j in range(m):
                if self.exsitWord(board, i, j, n, m, word):
                    return True
        return False

    def exsitWord(self, board, i, j, n, m, word):
        if not word:
            return True
        if i < 0 or i >= n or j < 0 or j >= m or board[i][j] != word[0]:
            return False
        board[i][j] = ' '
        sub = word[1:]
        result = self.exsitWord(board, i+1, j, n, m, sub) \
                 or self.exsitWord(board, i-1, j, n, m, sub) \
                 or self.exsitWord(board, i, j+1, n, m, sub) \
                 or self.exsitWord(board, i, j-1, n, m, sub)
        board[i][j] = word[0]
        return result

    #N-Queens
    def solveNQueens(self, n):
        stack, result = [[(0, i)] for i in range(n)], []
        while stack:
            board = stack.pop()
            row = len(board)
            if row == n:
                result.append([''.join('Q' if i == c else '.' for i in range(n)) for _, c in board])
            else:
                for col in range(n):
                    if all(col != c and abs(row-r) != abs(col-c) for r, c in board):
                        stack.append(board+[(row, col)])
        return result

    #N-Queens bit
    def solveNQueensii(self, n):
        limit = (1 << n) - 1
        result, each = [], []
        self.bittest(limit, 0, 0, 0, result, each)
        return result

    def bittest(self, limit, row, ld, rd, result, each):
        if row == limit:
            result.append(each.copy())
            return
        pos = limit & (~(row | ld | rd))
        while pos:
            p = pos & (-pos)
            pos -= p
            n = int(math.log2(limit+1))
            i = int(math.log2(p))
            each.append((n-i-1)*'.'+'Q'+i*'.')
            self.bittest(limit, row | p, (ld | p) << 1, (rd | p) >> 1, result, each)
            each.pop()

    #N-Queens II
    def totalNQueens(self, n):
        stack, count = [[(0, i)] for i in range(n)], 0
        while stack:
            board = stack.pop()
            row = len(board)
            if row == n:
                count += 1
            else:
                for col in range(n):
                    if all(col != c and abs(row-r) != abs(col-c) for r, c in board):
                        stack.append(board+[(row, col)])
        return count

    def totalNQueensii(self, n):
        limit = (1 << n) - 1
        result = [0]
        self.bittestii(limit, 0, 0, 0, result)
        return result[0]

    def bittestii(self, limit, row, ld, rd, result):
        if row == limit:
            result[0] += 1
            return
        pos = limit & (~(row | ld | rd))
        while pos:
            p = pos & (-pos)
            pos -= p
            self.bittestii(limit, row | p, (ld | p) << 1, (rd | p) >> 1, result)

    #Letter Combinations of a Phone Number
    def letterCombinations(self, digits):
        if not digits:
            return []
        kvmaps = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        return functools.reduce(lambda acc, digit: [x + y for x in acc for y in kvmaps[digit]], digits, [''])

    #Letter Combinations of a Phone Number DFS
    def letterCombinationsii(self, digits):
        if not digits:
            return []
        dict = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        result = []
        self.helper(digits, dict, result, '')
        return result

    def helper(self, digits, dict, result, temp):
        if not digits:
            result.append(temp)
            return
        d = digits[0]
        l = dict[d]
        for c in l:
            temp += c
            self.helper(digits[1:], dict, result, temp)
            temp = temp[:-1]

    #Largest Rectangle in Histogram
    #!!!
    def largestRectangleArea(self, height):
        length = len(height)
        if length < 2:
            return height[0] if length else 0
        stack, maxarea = [], 0
        for i in range(length):
            current = height[i]
            while stack:
                if current < stack[-1][1]:
                    index, h, leftarea = stack.pop()
                    maxarea = max(h*(i-index-1)+leftarea, maxarea)
                elif current == stack[-1][1]:
                    stack.pop()
                else:
                    break
            if stack:
                stack.append((i, current, current*(i-stack[-1][0])))
            else:
                stack.append((i, current, current*(i+1)))
        while stack:
            index, h, leftarea = stack.pop()
            maxarea = max(h*(length-index-1)+leftarea, maxarea)
        return maxarea

    #Largest Rectangle in Histogram
    #magic solution
    def largestRectangleAreaii(self, height):
        length = len(height)
        if length < 2:
            return height[0] if length else 0
        maxarea = 0
        height.append(0)
        for i in range(length+1):
            j = i - 1
            while j >= 0 and height[j] > height[i]:
                maxarea = max(height[j]*(i-j), maxarea)
                height[j] = height[i]
                j -= 1
        return maxarea

    #Unique Binary Search Trees II
    #https://leetcode.com/discuss/29330/brief-python-dp-solution
    def generateTrees(self, n):
        result = {0: [None]}
        for i in range(1, n + 1):
            result[i] = []
            for pos in range(1, i + 1):
                for left in result[pos - 1]:
                    for right in result[i - pos]:
                        root = TreeNode(pos)
                        result[i].append(root)
                        right = self.treesubtitute(right, range(pos + 1, i + 1))
                        root.left = left
                        root.right = right
        return result[n]

    def treesubtitute(self, root, nums):
        if not root:
            return None
        new = TreeNode(nums[root.val - 1])
        if root.left:
            new.left = self.treesubtitute(root.left, nums)
        if root.right:
            new.right = self.treesubtitute(root.right, nums)
        return new

    #Edit Distance
    def minDistance(self, word1, word2):
        n, m = len(word1), len(word2)
        if n * m == 0:
            return n or m
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        word1, word2 = ' ' + word1, ' ' + word2
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if word1[i] == word2[j] else 1
                dp[i][j] = min(dp[i-1][j-1] + cost, dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[n][m]

    #Palindrome Partitioning
    def partition(self, s):
        if not s:
            return []
        result = []
        self.dfspartition(s, result, [])
        return result

    def dfspartition(self, s, result, each):
        if not s:
            result.append(each[:])
            # An excellent way to copy a list
            return
        for i in range(1, len(s) + 1):
            sub = s[:i]
            if sub == sub[::-1]:
                each.append(sub)
                self.dfspartition(s[i:], result, each)
                each.pop()

    #Distinct Subsequences
    def numDistinct(self, s, t):
        n, m = len(s), len(t)
        if n < m:
            return 0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 1
        for j in range(1, m + 1):
            dp[0][j] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if i < j:
                    dp[i][j] = 0
                else:
                    if s[i-1] == t[j-1]:
                        dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                    else:
                        dp[i][j] = dp[i-1][j]
        return dp[n][m]