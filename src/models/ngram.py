import pickle
from collections import *
import numpy as np
import utils
import matplotlib.pyplot as plt


def normalize(counter):
    s = float(sum(counter.values()))
    for c,cnt in counter.iteritems():
        counter[c] = cnt/s
    return counter
    

def trainsingleorder(data, alph, order):
    if (order == 0):
        markov = Counter()
        for song in range(len(data)):
            for i in range(len(alph)):
                markov[i] = data[song].count(i)+1
        norm = normalize(markov)
        return norm

    markov = defaultdict(Counter)
    
    for song in data:
        for i in range(len(song)-order):
            history = "-".join(str(j) for j in song[i:i+order])
            curr = str(song[i+order])
            markov[history][curr]+=1
    
    norm = {hist:normalize(chars) for hist, chars in markov.iteritems()}
    return norm 

def flatten_data(data):
    """
    Transforms a set of songs defined by dictionaries ('dtseqs' etc...) to a single sequence of tuples (dT, T, pitch)
    """
    zipped = [[(dt,t,p) for dt,t,p in zip(dT,T,P)] for dT,T,P in zip(data["dtseqs"], data["Tseqs"], data["pitchseqs"])]
    return [x for X in zipped for x in X]

class Ngram:
    """
    Stores an Ngram model

    :attr order: an integer specifying the order (> 1)
    :attr alph: the alphabet of characters
    :attr probs: a list of dictionaries mapping from a history of length i (the index of the list)
                 to the probabilities of next character (another dictionary)
                 For index 0, just the dictionary of global probabilities of each character
    """

    def __init__(self, order):
        self.order = order
        self.probs = []

    def train(self, data, alph):
        """
        :param data: a single sequence of tuples (dT, T, pitch)
        :param alph: a set of tuples (dT, T, pitch)
        """
        self.alph = alph
        for i in range(self.order+1):
            self.probs.append(trainsingleorder(data, self.alph, i))

        return self.avg_loglikelihood_uniform(data)

    def generate(self, nletters = 250):
        probs = self.probs
        order = self.order
        out = []
        notes, chance = zip(*probs[0].items())
        notes = list(notes)
        chance = list(chance)
        out.append( np.random.choice( notes, p=chance ) )

        for i in range(order): 
            n = i
            history = "-".join(str(nb) for nb in out)
            while (history not in probs[n]) and (n > 0):
                n = n - 1
                history = "-".join(str(nb) for nb in out[i+(order-n):i+order])
            if (n == 0):
                notes, chance = zip(*probs[n].items())
                notes = list(notes)
                chance = list(chance)
                new = np.random.choice( notes, p=chance )
            else:
                notes, chance = zip(*probs[n][history].items())
                notes = list(notes)
                chance = list(chance)
                new = np.random.choice( notes, p=chance )
            out.append(new)

        for i in range(nletters-order):
            n = order
            history = "-".join(str(nb) for nb in out[i:i+order])
            while (history not in probs[n]) and (n > 0):
                n = n - 1
                history = "-".join(str(nb) for nb in out[i+(order-n):i+order])
            if (n == 0):
                notes, chance = zip(*probs[n].items())
                notes = list(notes)
                chance = list(chance)
                new = np.random.choice( notes, p=chance )
            else:
                notes, chance = zip(*probs[n][history].items())
                notes = list(notes)
                chance = list(chance)
                new = np.random.choice( notes, p=chance )
            out.append(new)
            #history = "-".join(str(nb) for nb in out[i+1:i+order+1])

        return out 

    def avg_loglikelihood_uniform(self, test):
        """
        :param test: a single sequence of tuples (dT, T, pitch)
        :returns: the log-likelihood of the sequence
        """

        res = 0
        order = self.order

        for i in range(len(test)-order):
            history = "-".join(str(nb) for nb in test[i:i+order])
            curr = str(test[i+order])
            if (order == 0):
                if curr in self.probs[0]:
                    res += np.log(self.probs[0][curr])
                else:
                    res += np.log(1./len(self.alph))
            if (history not in self.probs[order] or curr not in self.probs[order][history]):
                res += np.log(1.0/len(self.alph))
            else:
                res += np.log(self.probs[order][history][curr])

        return res/(len(test)-order)


def avg_loglikelihood_fallback(training,alph,order,test):
    res = 0
    fails = 0

    probs = []
    for i in range(order+1):
        probs.append(train(training, alph, i))

    for song in test:
        res_song = 0

        for i in range(len(song)-order):
            n = order
            history = "-".join(str(nb) for nb in song[i:i+order])
            curr = str(song[i+order])
            while (n > 0) and (history not in probs[n] or curr not in dict(probs[n][history])):
                n = n - 1
                history = "-".join(str(nb) for nb in song[i+(order-n):i+order])
            if (n == 0):
                if (probs[n][song[i+order]] != 0):
                    res_song += np.log(probs[n][song[i+order]])
                else:
                    print "warning : symbol nb ", song[i+order], "not found"
                    fails += 1
            else:
                res_song += np.log(probs[n][history][curr])
        res += res_song/(len(song)-order)

    return res/len(test)

def avg_loglikelihood_uniform(alph,order,test):

    res = 0
    fails = 0

    probs = train(training,alph,order)

    for song in test:
        res_song = 0

        for i in range(len(song)-order):
            history = "-".join(str(nb) for nb in song[i:i+order])
            curr = str(song[i+order])
            if (order == 0):
                res_song += np.log(probs[song[i+order]])
            if (history not in probs or curr not in dict(probs[history])):
                res_song += np.log(1.0/len(alph))
            else:
                res_song += np.log(probs[history][curr])
        res += res_song/(len(song)-order)

    return res/len(test)

# if __name__ == '__main__':
    # datafile = "data/data.pkl"
    # f = pickle.load(open(datafile,"rb"))

    # degree = 9

    # rep = 'generated/'

    # 12 tones representation

    # data12tones = f[0]['RhythmMelody'][:int(len(f[0]['RhythmMelody'])*.8)]
    # test12tones = f[0]['RhythmMelody'][int(len(f[0]['RhythmMelody'])*.8):]
    # alph12tones = f[1]['RhythmMelody']

    # probs12tones = []
    # for i in range(degree):
    #   probs12tones.append(train(data12tones, alph12tones, i))

    # song12tones = [[int(s) for s in generate(probs12tones, alph12tones, degree)]]

    # label12tones = [['my song using 12 tones representation']]

    # name12tones = rep + '12tones/' + str(degree) + '-gram.abc'
    # abc_out = open(name12tones, 'w')
    # utils.write_abc_12TonesRepr(song12tones, alph12tones, label12tones, abc_out)
    # abc_out.close()

    # regular split representation
    # data_rhythm = f[0]['Rhythm']
    # alph_rhythm = f[1]['Rhythm']
    # data_melody = f[0]['Melody']
    # alph_melody = f[1]['Melody']

    # probs_rhythm = []
    # for i in range(degree):
    #   probs_rhythm.append(train(data_rhythm, alph_rhythm, i))

    # probs_melody = []
    # for i in range(degree):
    #   probs_melody.append(train(data_melody, alph_melody, i))

    # song_rhythm = [[int(s) for s in generate(probs_rhythm, alph_rhythm, degree)]]
    # song_melody = [[int(s) for s in generate(probs_melody, alph_melody, degree)]]

    # label = [['my song using split representation']]

    # name = rep + 'split/' + str(degree) + '-gram.abc'
    # abc_out = open(name, 'w')
    # utils.write_abc_from_state_vec(song_melody, alph_melody, song_rhythm, alph_rhythm, label, abc_out)
    # abc_out.close()

    # training_error = []
    # training_error.append(np.log(1.0/len(alph12tones)))
    # for i in range(degree):
    #   training_error.append(avg_loglikelihood(probs12tones,i,data12tones))

    # test_error = []
    # test_error.append(np.log(1.0/len(alph12tones)))
    # for i in range(degree):
    #   test_error.append(avg_loglikelihood(probs12tones,i,test12tones))

    # plt.xlim(0,5)
    # plt.ylim(-10,0)
    # plt.plot(training_error, 'o-g', label='training error')
    # plt.plot(test_error, 'o-b', label='test error')
    # plt.legend(loc=1, borderaxespad=1, fontsize='small')
    # plt.xticks(np.arange(degree+1), ('uniform', '0-gram', '1-gram', '2-gram', '3-gram', '4-gram', '5-gram', '6-gram', '7-gram', '8-gram', '9-gram', '10-gram', '11-gram', '12-gram', '13-gram', '14-gram', '15-gram', '16-gram', '17-gram', '18-gram', '19-gram', '20-gram'), rotation=70, fontsize=8)
    # plt.savefig('log-likelihood-music-12tones')
    # plt.close()