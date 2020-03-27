# -*- coding: utf-8 -*-


def removeDuplicates(self, head):
    _cursor = head
    while _cursor:
        while _cursor.next and _cursor.data == _cursor.next.data:
            _cursor.next = _cursor.next.next
        _cursor = _cursor.next
    return head

