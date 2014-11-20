__author__ = 'mengpeng'


def preorder(root):
    if not root:
        return []
    stack = []
    result = []
    node = root
    while node or stack:
        if node:
            result.append(node.val)
            if node.right:
                stack.append(node.right)
            node = node.left
        else:
            node = stack.pop()
    return result


def preorder_my(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result