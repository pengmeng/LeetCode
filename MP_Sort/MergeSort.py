__author__ = 'mengpeng'


def merge(array, p, q, r):
    left, right = array[p:q+1], array[q+1:r]
    lenl, lenr = len(left), len(right)
    i, j, k = 0, 0, p
    while i < lenl and j < lenr:
        if left[i] < right[j]:
            array[k] = left[i]
            i += 1
        else:
            array[k] = right[j]
            j += 1
        k += 1
    while i < lenl:
        array[k] = left[i]
        i += 1
        k += 1
    while j < lenr:
        array[k] = right[j]
        j += 1
        k += 1


def merge_sort(array, start, end):
    if start < end:
        m = (start + end) // 2
        merge_sort(array, start, m)
        merge_sort(array, m+1, end)
        merge(array, start, m, end)