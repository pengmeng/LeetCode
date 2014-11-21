__author__ = 'mengpeng'


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


def quicksort_list(head, end):
    if head and head is not end:
        par = partition_list(head, end)
        quicksort_list(head, par)
        quicksort_list(par.next, end)


def partition_list(head, end):
    pivot = head.val
    i = head
    j = head.next
    while j is not end:
        if j.val <= pivot:
            i = i.next
            i.val, j.val = j.val, i.val
        j = j.next
    head.val, i.val = i.val, head.val
    return i