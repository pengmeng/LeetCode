__author__ = 'mengpeng'


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        self.tail = None

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
        head.tail = pred
        return head

    def __str__(self):
        head = self
        result = []
        while head:
            result.append(head.val)
            head = head.next
        return result.__str__()

    #add exception with invalid input
    def __getitem__(self, item):
        if item < 0:
            return None
        index, head = item, self
        while index > 0 and head:
            head = head.next
            index -= 1
        return head

    def __gt__(self, other):
        return self.val > other

    def __lt__(self, other):
        return self.val < other

    def show(self):
        print(self.toarray())

    def toarray(self):
        head = self
        result = []
        while head:
            result.append(head.val)
            head = head.next
        return result


class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
