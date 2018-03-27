from utils.midiparser import getDictionaries, parseFolder, cleanDic, writeMIDI

def preprocess(datapath, writeMIDI=False):
    """
    Parses a datapath and returns a preprocessed dataset, either splitted or not
    :param datapath:
    :param writeMIDI:
    :return:
    """
    dictionaries = getDictionaries()
    dataset = parseFolder(datapath, dictionaries)
    dictionaries = cleanDic(dataset, dictionaries)
    xdTs, xTs, xPs, dTvocsize, Tvocsize, pitchvocsize = toZ(dataset, dictionaries)
    dataset = {'dTseqs': xdTs, 'tseqs': xTs, 'pitchseqs': xPs}

    if writeMIDI:
        dtseq, Tseq, pitchseq = toMIDI(xdTs[0], xTs[0], xPs[0], dictionaries)
        for dt, t, p in zip(dtseq, Tseq, pitchseq):
            print(dictionaries['duration_text'][dictionaries['dtseqs'].index(dt)],
                  dictionaries['duration_text'][dictionaries['Tseqs'].index(t)],
                  dictionaries['pitch_text'][dictionaries['pitchseqs'].index(p)])
        writeMIDI(dtseq, Tseq, pitchseq, path='../data/', label='example')

    return dataset, (dTvocsize, Tvocsize, pitchvocsize), dictionaries


def split(dataset, k):
    """
    :param dataset: a dictionary of list of sequences containing the lists 'dtseqs', 'Tseqs', 'pitchseq'
    :param k: the number of folds
    :return: a list of dictionaries {'train':, 'test':} where train and test have the same format as dataset
    (note: train and test are views, and not copies of the original dataset)
    """
    n = len(dataset['tseqs'])
    ret = []
    for i in xrange(k):
        ret.append({'train':{}, 'test':{}})
        for key in dataset:
            ret[-1]['train'][key] =  dataset[key][:i*(n//k)] + dataset[key][min((i+1)*(n//k), n):]
            ret[-1]['test'][key] = dataset[key][i*(n//k):min((i+1)*(n//k), n)]
    return ret



def transpose(pitch_seq):
    upperbound = max(pitch_seq)
    lowerbound = min(pitch_seq)
    possible_shifts = range(0 - lowerbound, pitchvocsize - upperbound)
    if len(possible_shifts) == 0:
        possible_shifts.append(0)
    shift = np.random.choice(possible_shifts)
    out = [x + shift for x in pitch_seq]
    return out


def toZ(data, dictionaries):
    dTvocsize = len(dictionaries["dtseqs"])
    Tvocsize = len(dictionaries["Tseqs"])
    pitchvocsize = len(dictionaries["pitchseqs"])

    ## Translate

    xdT = []
    xP = []
    xT = []
    for dTseq, Tseq, pitchseq in zip(data["dtseqs"], data["Tseqs"], data["pitchseqs"]):
        dTs = [dictionaries["dtseqs"].index(x) for x in dTseq]
        Ts = [dictionaries["Tseqs"].index(x) for x in Tseq]
        vs = [dictionaries["pitchseqs"].index(x) for x in pitchseq]

        xdT.append(dTs)
        xT.append(Ts)
        xP.append(vs)

    return xdT, xT, xP, dTvocsize, Tvocsize, pitchvocsize


def toMIDI(xdT, xT, xP, dictionaries):
    dT = [dictionaries["dtseqs"][x] for x in xdT]
    T = [dictionaries["Tseqs"][x] for x in xT]
    P = [dictionaries["pitchseqs"][x] for x in xP]

    return dT, T, P


def sortbylength(data):
    idxes = np.argsort([len(x) for x in data["dtseqs"]])
    data["dtseqs"] = list(np.asarray(data["dtseqs"])[idxes])
    data["Tseqs"] = list(np.asarray(data["Tseqs"])[idxes])
    data["pitchseqs"] = list(np.asarray(data["pitchseqs"])[idxes])

    return data


def augment(xdTs, xTs, xPs):
    for xdT, xT, xP in zip(xdTs, xTs, xPs):
        transpose(xP)
