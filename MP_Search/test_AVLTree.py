__author__ = 'mengpeng'
from unittest import TestCase
from MP_Search.AVLTree import AVLTree
from MP_Search.BTTraversal import inorder
from random import randint
from LeetSolution import LeetSolution


class TestAVLTree(TestCase):
    def test_fromlist(self):
        s = LeetSolution()
        l = [randint(0, 100) for x in range(0, 10)]
        l = list(set(l))
        avl = AVLTree.fromlist(l)
        l.sort()
        self.assertEqual(l, inorder(avl.root))
        #print(s.levelOrder(avl.root))

    def test_search(self):
        s = LeetSolution()
        l = [randint(0, 100) for x in range(0, 10)]
        l = list(set(l))
        avl = AVLTree.fromlist(l)
        i = randint(0, len(l) - 1)
        self.assertEqual(l[i], avl.search(l[i]).val)
        self.assertIsNone(avl.search(1000))
        self.assertIsNone(avl.search(-100))

    def test_minmax(self):
        l = [randint(0, 100) for x in range(0, 10)]
        l = list(set(l))
        avl = AVLTree.fromlist(l)
        self.assertEqual(min(l), avl.getmin().val)
        self.assertEqual(max(l), avl.getmax().val)

    def test_delete(self):
        l = [randint(0, 100) for x in range(0, 10)]
        l = list(set(l))
        avl = AVLTree.fromlist(l)
        l.sort()
        i = randint(0, len(l) - 1)
        ele = l[i]
        l.remove(ele)
        avl.delete(ele)
        self.assertEqual(l, inorder(avl.root))
        self.assertRaises(KeyError, avl.delete, 1000)
        self.assertRaises(KeyError, avl.delete, -100)