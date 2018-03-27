from utils.tools import trainsingleorder, dic_argmax, dic_sample
from utils.midiparser import writeMIDI
from utils.preprocessing import toMIDI


class SingleNgram:
    """
    Simple fallback Ngram model on single sequence type
    """

    def __init__(self, order):
        self.order = order
        self.probs = []

    def train(self, data):
        for i in xrange(self.order+1):
            self.probs.append(trainsingleorder(data, i))

    def predictValue(self, hist):
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

    def generate(self, N):
        ret = []
        for i in range(N):
            hist = ret[max(0,i-self.order):i]
            distr = self.predictValue(hist)
            ret.append(dic_sample(distr))
        return ret


class INgram:
    """
    Ngram model assuming independence of sequences dT, t, and pitch
    :attr order: the order
    :attr models: a dictionary of underlying SingleNgram models for each sequence type
    """

    def __init__(self, order):
        self.order = order
        self.models = {'dTseqs': SingleNgram(order), 'tseqs': SingleNgram(order), 'pitchseqs': SingleNgram(order)}

    def train(self, data):
        """
        :param data: dataset
        """
        for key in self.models:
            self.models[key].train(data[key])
        return self.predict(data)

    def predict(self, data):
        """
        :param data:
        :return:
        """
        predictions = {"dTseqs": [], "tseqs": [], "pitchseqs": []}
        for s in range(len(data["dTseqs"])):
            predictions["dTseqs"].append([])
            predictions["tseqs"].append([])
            predictions["pitchseqs"].append([])

            for i in range(len(data["dTseqs"][s])):
                for key in ("dTseqs", "tseqs", "pitchseqs"):
                    hist = data[key][s][max(0,i-self.order):i]
                    predictions[key][s].append(self.models[key].predictValue(hist))

        return predictions

    def generate(self, n_songs=20, N=200, write_MIDI=False, dictionaries=None, path='data/generated'):
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
            song = self.generate_song(N)
            dataset['dTseqs'].append(song['dTseqs'])
            dataset['tseqs'].append(song['tseqs'])
            dataset['pitchseqs'].append(song['pitchseqs'])
            if write_MIDI:
                dtseq, tseq, pseq = toMIDI(song['dTseqs'], song['tseqs'], song['pitchseqs'], dictionaries)
                writeMIDI(dtseq, tseq, pseq, path=path,
                          label='ingram-order' + str(self.order) + '-' + str(i))
        return dataset

    def generate_song(self, N=200):
        return {"dTseqs": self.models["dTseqs"].generate(N), "tseqs": self.models['tseqs'].generate(N),
                'pitchseqs': self.models['pitchseqs'].generate(N)}

