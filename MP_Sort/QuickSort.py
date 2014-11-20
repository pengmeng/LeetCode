__author__ = 'mengpeng'

from ListNode import ListNode


def quicksort_array(array, start, end):
    if start < end:
        par = partition_array(array, start, end)
        quicksort_array(array, start, par - 1)
        quicksort_array(array, par + 1, end)


def partition_array(array, start, end):
    pivot = array[end]
    i = start - 1
    for j in range(start, end):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i+1], array[end] = array[end], array[i+1]
    return i+1


def quicksort_list(head):
    pass


def partition_list(head, end):
    pivot = head.val
    curr = head
    pred = ListNode()
    pred.next = head
    while curr.next is not end:
        if curr.next.val <= pivot:
            exchangenode(pred, curr)
            pred = pred.next


def exchangenode(pred1, pred2):
    pass
