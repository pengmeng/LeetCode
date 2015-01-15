from unittest import TestCase
from TreeNode import TreeNode
import MP_Search.BTTraversal as BTTraversal

__author__ = 'mengpeng'


class TestPreorder(TestCase):
    def test_preorder(self):
        self.assertEqual([], BTTraversal.preorder([]))
        root = TreeNode.makeTree([1, '#', 2, 3])
        self.assertEqual([1, 2, 3], BTTraversal.preorder(root))
        root = TreeNode.makeTree([1, 2, 2, '#', 3, '#', 3])
        self.assertEqual([1, 2, 3, 2, 3], BTTraversal.preorder(root))
        root = TreeNode.makeTree([3, 1, 2])
        self.assertEqual([3, 1, 2], BTTraversal.preorder(root))

    def test_preorder_my(self):
        self.assertEqual([], BTTraversal.preorder_my([]))
        root = TreeNode.makeTree([1, '#', 2, 3])
        self.assertEqual([1, 2, 3], BTTraversal.preorder_my(root))
        root = TreeNode.makeTree([1, 2, 2, '#', 3, '#', 3])
        self.assertEqual([1, 2, 3, 2, 3], BTTraversal.preorder_my(root))
        root = TreeNode.makeTree([3, 1, 2])
        self.assertEqual([3, 1, 2], BTTraversal.preorder_my(root))

    def test_inorder(self):
        root = TreeNode.makeTree([1, '#', 2, 3])
        self.assertEqual([1, 3, 2], BTTraversal.inorder(root))
        root = TreeNode.makeTree([1, 2, 3, 4, 5, '#', 6])
        self.assertEqual([4, 2, 5, 1, 3, 6], BTTraversal.inorder(root))
        root = TreeNode.makeTree([3, 1, 4, '#', 2, '#', 5, '#', '#', '#', 6])
        self.assertEqual([1, 2, 3, 4, 5, 6], BTTraversal.inorder(root))

    def test_postorder(self):
        self.assertEqual([], BTTraversal.postorder([]))
        root = TreeNode.makeTree([1, '#', 2, 3])
        self.assertEqual([3, 2, 1], BTTraversal.postorder(root))
        root = TreeNode.makeTree([1, 2, 2, '#', 3, '#', 3])
        self.assertEqual([3, 2, 3, 2, 1], BTTraversal.postorder(root))
        root = TreeNode.makeTree([3, 1, 2])
        self.assertEqual([1, 2, 3], BTTraversal.postorder(root))