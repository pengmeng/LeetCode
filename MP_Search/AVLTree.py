__author__ = 'mengpeng'


class AVLNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 0


class AVLTree(object):
    def __init__(self):
        self.root = None

    @staticmethod
    def fromlist(l):
        avl = AVLTree()
        for each in l:
            avl.insert(each)
        return avl

    def search(self, key):
        return self._search(key, self.root) if self.root else None

    def _search(self, key, node):
        if not node:
            return None
        elif key < node.val:
            return self._search(key, node.left)
        elif key > node.val:
            return self._search(key, node.right)
        else:
            return node

    def getmin(self):
        return self._getmin(self.root) if self.root else None

    def _getmin(self, node):
        return self._getmin(node.left) if node.left else node

    def getmax(self):
        return self._getmax(self.root) if self.root else None

    def _getmax(self, node):
        return self._getmax(node.right) if node.right else node

    def height(self, node):
        return node.height if node else -1

    def updateheight(self, node):
        node.height = max(self.height(node.left), self.height(node.right)) + 1

    #rotation
    def rightrotate(self, node):
        k = node.left
        node.left = k.right
        k.right = node
        node.height = max(self.height(node.left), self.height(node.right)) + 1
        k.height = max(self.height(k.left), node.height) + 1
        return k

    def leftrotate(self, node):
        k = node.right
        node.right = k.left
        k.left = node
        node.height = max(self.height(node.left), self.height(node.right)) + 1
        k.height = max(node.height, self.height(node.right)) + 1
        return k

    def leftrightrotate(self, node):
        node.left = self.leftrotate(node.left)
        return self.rightrotate(node)

    def rightleftrotate(self, node):
        node.right = self.rightrotate(node.right)
        return self.leftrotate(node)

    #insertion
    def insert(self, key):
        self.root = self._insert(key, self.root) if self.root else AVLNode(key)

    def _insert(self, key, node):
        if not node:
            node = AVLNode(key)
        elif key < node.val:
            node.left = self._insert(key, node.left)
            if self.height(node.left) - self.height(node.right) > 1:
                if key < node.left.val:
                    node = self.rightrotate(node)
                else:
                    node = self.leftrightrotate(node)
        elif key > node.val:
            node.right = self._insert(key, node.right)
            if self.height(node.right) - self.height(node.left) > 1:
                if key > node.right.val:
                    node = self.leftrotate(node)
                else:
                    node = self.rightleftrotate(node)
        node.height = max(self.height(node.left), self.height(node.right)) + 1
        return node

    #deletion
    def delete(self, key):
        self.root = self._delete(key, self.root)

    def _delete(self, key, node):
        if not node:
            raise KeyError
        elif key < node.val:
            node.left = self._delete(key, node.left)
            if self.height(node.right) - self.height(node.left) > 1:
                if self.height(node.right.right) >= self.height(node.right.left):
                    node = self.leftrotate(node)
                else:
                    node = self.rightleftrotate(node)
            self.updateheight(node)
        elif key > node.val:
            node.right = self._delete(key, node.right)
            if self.height(node.left) - self.height(node.right) > 1:
                if self.height(node.left.left) >= self.height(node.left.right):
                    node = self.rightrotate(node)
                else:
                    node = self.leftrightrotate(node)
            self.updateheight(node)
        elif node.left and node.right:
            if node.left.height <= node.right.height:
                rightmin = self._getmin(node.right)
                node.val = rightmin.val
                node.right = self._delete(node.val, node.right)
            else:
                leftmax = self._getmax(node.left)
                node.val = leftmax.val
                node.left = self._delete(node.val, node.left)
            self.updateheight(node)
        else:
            node = node.right if node.right else node.left
        return node