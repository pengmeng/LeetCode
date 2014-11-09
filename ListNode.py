__author__ = 'mengpeng'


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    @staticmethod
    def makeList(array):
        if not array:
            return None
        head = ListNode(array.pop(0))
        pred = head
        while array:
            item = ListNode(array.pop(0))
            pred.next = item
            pred = item
        return head

    def show(self):
        head = self
        result = []
        while head:
            result.append(head.val)
            head = head.next
        print(result)