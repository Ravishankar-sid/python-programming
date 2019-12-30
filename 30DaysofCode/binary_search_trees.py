    def getHeight(self,root):
        if root == None or root.left == root.right == None:
            return 0 
        else:
            return 1+ max(self.getHeight(root.left), self.getHeight(root.right))
