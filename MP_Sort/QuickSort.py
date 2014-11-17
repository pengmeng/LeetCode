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