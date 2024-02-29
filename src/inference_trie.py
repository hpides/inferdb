from collections import Counter

class TrieNode:
    """A node in the trie structure"""

    def __init__(self):
        # the partial key stored in this node
        # self.prefix = str(key)

        # a list of child nodes
        self.children = {}


class TrieLeaf:
    """A leaf in the trie structure"""

    def __init__(self):
        # the character stored in this node
        # self.prefix = str(key)

        # values
        self.value = 0


class Trie(object):
    """The trie object"""

    def __init__(self, type):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode()
        self.type = type

    def insert(self, key, value):
        """Insert a key into the trie"""
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for idx, i in enumerate(key):
            i = int(i)
            if i in node.children:
                node = node.children[i]
            else:
                # If a character is not found,
                # create a new node in the trie
                if idx < len(key) - 1:
                    # new_node = TrieNode(node.prefix + str(i))
                    new_node = TrieNode()
                    node.children[i] = new_node
                    node = new_node
                # If we reach the end of the key, create a leaf and add the reference
                else:
                    new_leaf = TrieLeaf()
                    new_leaf.value = value
                    node.children[i] = new_leaf

    def dfs(self, node):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
        """
        if any(isinstance(children, TrieLeaf) for children in node.children.values()):
            for children in node.children.values():
                self.output.append(children.value)
        else:
            for child in node.children.values():
                self.dfs(child)

    def query(self, x):
        """Finds a key in the structure or performs prefix aggregation to create a prediction

        Args:
            x (list, array): list or array containing the keys to lookup

        Returns:
            int, float: value for the key in the structure
        """        
        node = self.root

        for k in x:
            if k in node.children:
                node = node.children[k]
                if isinstance(node, TrieLeaf):
                    return node.value
                else:
                    continue
            else:
                self.output = []
                self.dfs(node)
                if self.type == 'regression':
                    return sum(self.output) / len(self.output)
                elif self.type in ('classification', 'multi-class'):
                    counter = Counter(self.output)
                    return counter.most_common(1)[0][0]