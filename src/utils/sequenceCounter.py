
def metric_compare(corpus1, corpus2, levels=[2, 4, 8, 16, 32]):
    dic_tree = DictionaryTree(corpus1, max_level=max(levels))
    return dic_tree.countSequences(corpus2, levels)


class DictionaryTree:
    """
    :attr root: TreeNode
    """

    def __init__(self, corpus, max_level):
        """
        Create a new DictionaryTree summarizing a corpus
        """
        self.root = TreeNode(0)
        whole_corpus = isinstance(corpus, dict)
        num_songs = len(corpus["dTseqs"]) if whole_corpus else len(corpus)

        for s in range(num_songs):
            rolling_writers = [self.root]
            n = len(corpus["dTseqs"][s]) if whole_corpus else len(corpus[s])

            for i in range(n):
                if whole_corpus:
                    next_note = (corpus["dTseqs"][s][i], corpus["tseqs"][s][i], corpus["pitchseqs"][s][i])
                else:
                    next_note = corpus[s][i]

                # add the next note to the tree and let the writers go down one level
                rolling_writers = [writer.add_child_and_move(next_note) for writer in rolling_writers]
                # remove the oldest writer if it hits the last level
                if len(rolling_writers) == max_level:
                    rolling_writers = rolling_writers[1:]
                # append a new writer
                rolling_writers.append(self.root)

    def countSequences(self, corpus, levels):
        """
        For each sequence length in levels, count the proportions of sequences of this length in corpus that
        are also in this DictionaryTree
        :return: a dict ( level -> frequency of occurrence )
        """
        counters = {lvl: 0 for lvl in levels}
        num_seqs = {lvl: 0 for lvl in levels}  # the number of sequences of each length in the corpus
        whole_corpus = isinstance(corpus, dict)
        num_songs = len(corpus["dTseqs"]) if whole_corpus else len(corpus)

        for s in range(num_songs):
            # compute the number of sequences for each length
            n = len(corpus["dTseqs"][s]) if whole_corpus else len(corpus[s])
            for lvl in num_seqs:
                num_seqs[lvl] += n - lvl + 1

            # Go through the song counting the occurrences of sequences in the tree
            rolling_readers = [self.root]
            for i in range(n):
                if whole_corpus:
                    next_note = (corpus["dTseqs"][s][i], corpus["tseqs"][s][i], corpus["pitchseqs"][s][i])
                else:
                    next_note = corpus[s][i]

                # move the readers down the tree if possible
                rolling_readers = [reader.move(next_note) for reader in rolling_readers]
                # keep only the readers that could go down, and add a new one at the root
                rolling_readers = [reader for reader in rolling_readers if reader is not None] + [self.root]
                # increase counters when appropriate
                for reader in rolling_readers:
                    if reader.depth in levels:
                        counters[reader.depth] += 1
                # remove oldest reader if appropriate
                if rolling_readers[0].depth == max(levels):
                    rolling_readers = rolling_readers[1:]

        # divide number of occurrences by number of sequences
        return {lvl: counters[lvl]/float(num_seqs[lvl]) for lvl in counters}


class TreeNode:
    """
    :attr children: dict( note -> TreeNode )  note can be anything, altough it will usually be a triple (dT, T, p)
    :attr depth: int
    """

    def __init__(self, depth):
        self.depth = depth
        self.children = dict()

    def add_child_and_move(self, note):
        if note not in self.children:
            self.children[note] = TreeNode(self.depth+1)
        return self.children[note]

    def move(self, note):
        if note in self.children:
            return self.children[note]
        else:
            return None


"""
What follows is for testing purposes
"""
def gen_corpus(string):
    corpus = {"dTseqs": [[]], "tseqs": [[]], "pitchseqs": [[]]}
    for c in string:
        corpus["dTseqs"][0].append(ord(c))
        corpus["tseqs"][0].append(0)
        corpus["pitchseqs"][0].append(0)
    return corpus


def test():
    corpus1 = gen_corpus('bibliography')
    corpus2 = gen_corpus('bibliotheque')
    print(metric_compare(corpus1, corpus2, levels=[2, 3, 4, 6]))
