class Node:
    """
    Represents a node in the search tree.
    """
    def __init__(self, currentstate, parent, action):
        self.currentstate = currentstate
        self.parent = parent
        self.action = action

    def get_parent(self):
        return self.parent

    def get_currentstate(self):
        return self.currentstate

    def get_action(self):
        return self.action

    def reconstruct_path(self):
        """
        Reconstruct the path from this node to the root node.
        """
        path = []
        current = self
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
        return path[::-1]