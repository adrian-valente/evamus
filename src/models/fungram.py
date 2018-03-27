from utils.tools import trainsingleorder, dic_argmax, dic_sample, keys_subtract
from utils.preprocessing import toMIDI
from utils.midiparser import writeMIDI


class FUNgram:
    """
    Stores a Feature Unrolled Ngram model

    :attr tstart: first index of t values sector
    :attr pstart: first index of pitch values sector
    :attr nValues: total number of values in the 3 sectors (thus a value is in [0..nValues)
    :attr order: order of the model (0 for a simple frequency counter, 1 for normal Markov)
    :attr probs: sequence of fallback Ngram models (for 0 a dict {str(value) -> prob of occurrence}
                 for bigger orders a dict {str(history) -> {str(value) -> prob of occurrence} })
    """

    def __init__(self, order, sizes=(25,25,54)):
        self.tstart = sizes[0]
        self.pstart = self.tstart + sizes[1]
        self.nValues = self.pstart + sizes[2]
        self.order = order
        self.probs = []

    def preprocess(self, dataset):
        """
        Flattens the dataset in feature unrolled format
        Assumes the dataset is in Z format (integer values starting at zero for each sequence)
        :return: a list of sequences of unrolled notes (dT0, t0, p0, dT1, ....)
        """
        unrolled = []
        for dTs, ts, ps in zip(dataset['dTseqs'], dataset['tseqs'], dataset['pitchseqs']):
            unrolled.append([])
            for i in xrange(len(dTs)):
                unrolled[-1].append(dTs[i])
                unrolled[-1].append(self.tstart + ts[i])
                unrolled[-1].append(self.pstart + ps[i])
        return unrolled

    def train(self, dataset):
        """
        :param dataset: Standard format dataset
        """
        data = self.preprocess(dataset)
        for i in range(self.order+1):
            self.probs.append(trainsingleorder(data, i))
        return self.predict(dataset)

    def predict(self, dataset):
        """
        Returns predictions over a dataset
        :param dataset:
        :return:
        """
        predictions = {"dTseqs": [], "tseqs": [], "pitchseqs": []}
        data = self.preprocess(dataset)
        for song in data:
            predictions["dTseqs"].append([])
            predictions["tseqs"].append([])
            predictions["pitchseqs"].append([])
            l = len(song)
            for i in range(0, l, 3):
                hist = song[max(0,i-self.order):i]
                note = (song[i], song[i+1], song[i+2])
                pdT, pt, pp = self.predictNote(hist, note)
                predictions["dTseqs"][-1].append(pdT)
                predictions["tseqs"][-1].append(pt)
                predictions["pitchseqs"][-1].append(pp)
        return predictions

    def predictNote(self, hist, note):
        """
        Retuns 3 probability vectors, one for each component of the note
        :param hist: (a sequence) the unrolled history of previous notes
        :param note: (a triplet) the note (dT, t, pitch)
        :return: The 3 probability vectors (as defaultdicts)
        """
        dT, t, p = note
        pdT = self.predictValue(hist)
        hist.append(dT)
        pt = keys_subtract(self.predictValue(hist), self.tstart)
        hist.append(t)
        pp = keys_subtract(self.predictValue(hist), self.pstart)
        return pdT, pt, pp

    def predictValue(self, hist):
        """
        Gives probability distribution of next value given hist
        :param hist: an unrolled history of previous values
        :return: a defaultdict giving the probability distribution
        """
        if len(hist) > self.order:
            hist = hist[-self.order:]
        # Fallback part
        l = len(hist)
        while l > 0 and str(hist) not in self.probs[l]:
            l -= 1
            hist = hist[1:]

        # Retrieve predictions
        if l == 0:
            return self.probs[0]
        else:
            return self.probs[l][str(hist)]

    def generate(self, n_songs=20, N=200, write_MIDI=False, dictionaries=None, path='../data/generated/'):
        """
        Generate a dataset of songs
        :param n_songs: number of songs
        :param N: length of a song
        :param write_MIDI: write the songs to disk
        :param dictionaries: mapping from integers to note values, has to be set if write_MIDI is True
        :return:
        """
        dataset = {'dTseqs': [], 'tseqs': [], 'pitchseqs': []}
        for i in range(n_songs):
            song = self.postprocess(self.generate_song(N*3))
            dataset['dTseqs'].append(song['dTseqs'])
            dataset['tseqs'].append(song['tseqs'])
            dataset['pitchseqs'].append(song['pitchseqs'])
            if write_MIDI:
                dtseq, tseq, pseq = toMIDI(song['dTseqs'], song['tseqs'], song['pitchseqs'], dictionaries)
                writeMIDI(dtseq, tseq, pseq, path=path,
                          label='fungram-order'+str(self.order)+'-'+str(i))

        return dataset

    def generate_song(self, N=600, seed=None):
        """
        DO NOT USE OUT OF THE CLASS: not a standard format
        """
        assert N%3 == 0
        if seed is not None:
            ret = seed[:]
        else:
            ret = [0]

        for i in range(len(ret), N):
            hist = ret[max(0, i-self.order):]
            distr = self.predictValue(hist)
            ret.append(dic_sample(distr))
        return ret

    def postprocess(self, seq):
        """
        Converts from unrolled sequence format to a dictionary of 3 sequences (dTseqs, tseqs, pitchseqs)
        """
        song = {'dTseqs': [], 'tseqs': [], 'pitchseqs': []}
        for i,elt in enumerate(seq):
            if i%3 == 0:
                song['dTseqs'].append(elt)
            elif i%3 == 1:
                song['tseqs'].append(elt - self.tstart)
            elif i%3 == 2:
                song['pitchseqs'].append(elt - self.pstart)
        return song


