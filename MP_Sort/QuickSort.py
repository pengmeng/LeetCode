__author__ = 'mengpeng'


def quicksort_array(array, start, end):
    if start < end:
        par = partition_array(array, start, end)
        quicksort_array(array, start, par - 1)
        quicksort_array(array, par + 1, end)


def partition_array(array, start, end):
    pivot = array[end]
    i = start
    for j in range(start, end):
        if array[j] <= pivot:
            array[i], array[j] = array[j], array[i]
            i += 1
    array[i], array[end] = array[end], array[i]
    return i


def quicksort_3way(array, start, end):
    if start < end:
        l, r = partition_3way(array, start, end)
        quicksort_3way(array, start, l - 1)
        quicksort_3way(array, r + 1, end)


def partition_3way(array, start, end):
    pivot = array[end]
    l, i, r = start, start, end
    while i < r:
        if array[i] < pivot:
            array[i], array[l] = array[l], array[i]
            l += 1
            i += 1
        elif array[i] > pivot:
            r -= 1
            array[i], array[r] = array[r], array[i]
        else:
            i += 1
    array[r], array[end] = array[end], array[r]
    return l, r


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