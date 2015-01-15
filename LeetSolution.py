__author__ = 'mengpeng'
from ListNode import ListNode
import MP_Sort.InsertSort
import functools


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
        queue = [(root, 1)]
        result = []
        each = []
        cur_dep = 1
        while queue:
            (node, dep) = queue.pop(0)
            if not dep == cur_dep:
                cur_dep = dep
                result.append(each)
                each = []
            each.append(node.val)
            if node.left:
                queue.append((node.left, dep + 1))
            if node.right:
                queue.append((node.right, dep + 1))
        result.append(each)
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
        mid = (left + right) // 2
        while left != right:
            if num[mid] > num[right]:
                left = mid + 1
                mid = (left + right) // 2
            else:
                #ATTENTION! not right = mid - 1
                right = mid
                mid = (left + right) // 2
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
    def wordBreak(self, s, dict):
        if not s or not dict or s in dict:
            return s and s in dict
        length = len(s)
        subseq = [False for x in range(0, length + 1)]
        subseq[0] = True
        for i in range(1, length + 1):
            for j in range(0, i):
                if subseq[j]:
                    if s[j:i] in dict:
                        subseq[i] = True
                        break
        return subseq[length]