__author__ = 'mengpeng'
from ListNode import ListNode


def insertsort_list(head):
    if not head:
        return head
    pred = ListNode(0)
    pred.next = head
    curr = pred.next
    while curr.next:
        if curr.next.val < curr.val:
            ins = pred
            while ins.next.val < curr.next.val:
                ins = ins.next
            temp = curr.next
            curr.next = temp.next
            temp.next = ins.next
            ins.next = temp
        else:
            curr = curr.next
    return pred.next


def insertsort_array(array):
    for j in range(1, len(array)):
        key = array[j]
        i = j - 1
        while i >= 0 and array[i] > key:
            array[i+1] = array[i]
            i -= 1
        array[i+1] = key