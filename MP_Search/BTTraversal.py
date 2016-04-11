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
        if node:
            result.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return result


def inorder(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        while node:
            stack.append(node)
            node = node.left
        if stack:
            node = stack.pop()
            result.append(node.val)
            stack.append(node.right)
    return result


def postorder(root):
    if not root:
        return []
    stack = [root]
    pre = root
    result = []
    while stack:
        node = stack.pop()
        if (not node.left and not node.right) or pre is node.right or pre is node.left:
            result.append(node.val)
            pre = node
        else:
            stack.append(node)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
    return result