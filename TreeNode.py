__author__ = 'mengpeng'


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    @staticmethod
    def makeTree(array):
        array.insert(0, 0)
        length = len(array)
        root = TreeNode(array[1])
        queue = [root]
        index = 1
        while queue:
            node = queue.pop(0)
            if index * 2 < length:
                if array[index * 2] == '#':
                    node.left = None
                else:
                    node.left = TreeNode(array[index * 2])
            else:
                node.left = None
            if index * 2 + 1 < length:
                if array[index * 2 + 1] == '#':
                    node.right = None
                else:
                    node.right = TreeNode(array[index * 2 + 1])
            else:
                node.right = None
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            index += 1
        return root

    def showTree(self):
        if not self:
            print([])
        queue = [self]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        print(result)