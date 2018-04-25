"""
This file defines the tool PrefixTree, that can be used to compute the longest common subsequence
between two sequences.
To use:

pt = PrefixTree(song)
lcs = longest_common_subsequence(song)

Note that the input songs will be in standard format (a dict of "dTseqs", etc)
while the returned LCS will be a sequence of tuples (dT[i], T[i], P[i])
"""

class PrefixTree:
    """
    :attr root: TreeNode
    """

    def __init__(self, song):
        """
        Create a new DictionaryTree summarizing a corpus
        """
        self.root = TreeNode(-1, 0, None)

        rolling_writers = [self.root]
        n = len(song["dTseqs"])

        for i in range(n):
            next_note = (song["dTseqs"][i], song["tseqs"][i], song["pitchseqs"][i])
            # add the next note to the tree and let the writers go down one level
            rolling_writers = [writer.add_child_and_move(next_note) for writer in rolling_writers]
            # append a new writer
            rolling_writers.append(self.root)

    def longest_common_subsequence(self, song):
        rolling_readers = [self.root]
        n = len(song["dTseqs"])
        cur_deepest = rolling_readers[0]

        for i in range(n):
            next_note = (song["dTseqs"][i], song["tseqs"][i], song["pitchseqs"][i])
            # go down the tree if possible
            rolling_readers = [reader.move(next_note) for reader in rolling_readers]
            # remove readers that went to None
            rolling_readers = [r for r in rolling_readers if r is not None]
            # find if we went deeper
            if len(rolling_readers) > 0 and rolling_readers[0].depth > cur_deepest.depth:
                cur_deepest = rolling_readers[0]
            # append a new reader
            rolling_readers.append(self.root)

        # finally unroll the sequence backwards from the deepest reader
        l = []
        cur = cur_deepest
        while cur is not None:
            l.append(cur.value)
            cur = cur.parent

        del l[-1]
        return l[::-1]


class TreeNode:
    """
    :attr value: can be anything
    :attr children: dict (value -> child)
    :attr depth: int
    :attr parent: a TreeNode or Node if we are the root
    """

    def __init__(self, value, depth, parent):
        self.value = value
        self.depth = depth
        self.children = dict()
        self.parent = parent

    def add_child_and_move(self, note):
        if note not in self.children:
            self.children[note] = TreeNode(note, self.depth+1, self)
        return self.children[note]

    def move(self, note):
        if note in self.children:
            return self.children[note]
        else:
            return None
