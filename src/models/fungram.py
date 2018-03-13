from utils.tools import trainsingleorder
from math import log


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

    def __init__(self, sizes, order):
        self.tstart = sizes[0]
        self.pstart = self.tstart + sizes[1]
        self.nValues = self.pstart + sizes[2]
        self.order = order
        self.probs = []

    def preprocess(self, dataset):
        """
        Flattens the dataset in feature unrolled format
        Assumes the dataset is in Z format (integer values starting at zero for each sequence)
        :return: a single sequence of unrolled notes (dT0, t0, p0, dT1, ....)
        """
        unrolled = []
        for dTs, ts, ps in zip(dataset['dTseqs'], dataset['tseqs'], dataset['pitchseqs']):
            for i in xrange(len(dTs)):
                unrolled.append(dTs[i])
                unrolled.append(self.tstart + ts[i])
                unrolled.append(self.pstart + ps[i])
        return unrolled

    def train(self, data):
        """
        :param data: Unrolled dataset (apply preprocess first)
        """
        for i in range(self.order+1):
            self.probs.append(trainsingleorder(data, i))
        return self.test(data)

    def predict(self, hist):
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

    def probNote(self, hist, x):
        """
        :param hist: unrolled history
        :param x: a note x=(dT, t, pitch
        :return: the probability of note x given hist
        """
        dT, t, p = x
        pdT = self.predict(hist)[str(dT)]  # Pr(dT | hist)
        hist.append(dT)
        pt = self.predict(hist)[str(t)]   # Pr(t | dT, hist)
        hist.append(t)
        ppitch = self.predict(hist)[str(p)]  # Pr(p | t, dT, hist)
        return pdT * pt * ppitch

    def test(self, data, smoothing=False, smoothing_lambda=0.05):
        """
        Computes log-likelihood of sequence data (product of the probability of each note given previous ones)
        :param data: unrolled note sequence
        :return: log-likelihood
        """
        logL = 0.
        uniformP = 1./(self.tstart * (self.pstart-self.tstart) * (self.nValues-self.pstart))
        for i in xrange(0, len(data), 3):
            hist = data[max(0,i-self.order):i]
            x = (data[i], data[i+1], data[i+2])
            prob = self.probNote(hist, x)

            if smoothing:
                prob = smoothing_lambda * uniformP + (1 - smoothing_lambda) * prob

            if prob > 0:
                logL += log(prob)
            else:
                logL -= 100000

        return logL/len(data)
