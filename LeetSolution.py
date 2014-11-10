__author__ = 'mengpeng'
from TreeNode import *


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
        if last != 0:
            digits.insert(0, last)
        return digits