def levelOrder(self,root):
    _queue = [root if root else []]

    while _queue:
        _node = _queue.pop()
        print(_node.data, end=" ")

        if _node.left:
            _queue.insert(0,_node.left)
        if _node.right:
            _queue.insert(0,_node.right)
