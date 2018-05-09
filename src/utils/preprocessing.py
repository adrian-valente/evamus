from midiparser import getDictionaries, parseFolder, cleanDic, writeMIDI


def preprocess(datapath, writeMIDI=False, verbose=False):
    """
    Parses a datapath and returns a preprocessed dataset, along with essential information
    :param datapath: path to a folder containing MIDI files
    :param writeMIDI: if true, write preprocessed midi (single channel, discretized rythm) as midi files again
    :return: - the dataset in standard format
             - a triple (dTvocsize, Tvocsize, pitchvocsize) giving the size of the vocabulary for each sequence type,
             - a dictionary of mappings from Z to the corresponding pitches/rythms : it contains 5 sequences ('dTseqs',
             'Tseqs', 'duration_text', 'pitch_text', 'pitchseqs') such that 'dTseqs', 'tseqs' contain for each
             index in Z the corresponding rythm as decimal number and 'duration_text' is the same sequence with
             the rythms in fractional notation, and 'pitchseqs' contains for each integer in Z the corresponding pitch
             as a midi integer and 'pitch_text' is the same sequence with the pitches in musical notation.
    """
    dictionaries = getDictionaries()
    dataset, labels = parseFolder(datapath, dictionaries, verbose=verbose)
    dictionaries = cleanDic(dataset, dictionaries, verbose=verbose)
    xdTs, xTs, xPs, dTvocsize, Tvocsize, pitchvocsize = toZ(dataset, dictionaries)
    dataset = {'dTseqs': xdTs, 'tseqs': xTs, 'pitchseqs': xPs}

    if writeMIDI:
        dtseq, Tseq, pitchseq = toMIDI(xdTs[0], xTs[0], xPs[0], dictionaries)
        for dt, t, p in zip(dtseq, Tseq, pitchseq):
            print(dictionaries['duration_text'][dictionaries['dTseqs'].index(dt)],
                  dictionaries['duration_text'][dictionaries['tseqs'].index(t)],
                  dictionaries['pitch_text'][dictionaries['pitchseqs'].index(p)])
        writeMIDI(dtseq, Tseq, pitchseq, path='../data/', label='example')

    return dataset, (dTvocsize, Tvocsize, pitchvocsize), dictionaries, labels


def split(dataset, k):
    """
    :param dataset: a dictionary of list of sequences containing the lists 'dtseqs', 'tseqs', 'pitchseq'
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


# def transpose(pitch_seq):
#     upperbound = max(pitch_seq)
#     lowerbound = min(pitch_seq)
#     possible_shifts = range(0 - lowerbound, pitchvocsize - upperbound)
#     if len(possible_shifts) == 0:
#         possible_shifts.append(0)
#     shift = np.random.choice(possible_shifts)
#     out = [x + shift for x in pitch_seq]
#     return out


def toZ(data, dictionaries):
    dTvocsize = len(dictionaries["dTseqs"])
    Tvocsize = len(dictionaries["tseqs"])
    pitchvocsize = len(dictionaries["pitchseqs"])

    # Translate

    xdT = []
    xP = []
    xT = []
    for dTseq, Tseq, pitchseq in zip(data["dTseqs"], data["tseqs"], data["pitchseqs"]):
        dTs = [dictionaries["dTseqs"].index(x) for x in dTseq]
        Ts = [dictionaries["tseqs"].index(x) for x in Tseq]
        vs = [dictionaries["pitchseqs"].index(x) for x in pitchseq]

        xdT.append(dTs)
        xT.append(Ts)
        xP.append(vs)

    return xdT, xT, xP, dTvocsize, Tvocsize, pitchvocsize


def toMIDI(xdT, xT, xP, dictionaries):
    dT = [dictionaries["dTseqs"][x] for x in xdT]
    T = [dictionaries["tseqs"][x] for x in xT]
    P = [dictionaries["pitchseqs"][x] for x in xP]

    return dT, T, P


# def sortbylength(data):
#     idxes = np.argsort([len(x) for x in data["dTseqs"]])
#     data["dTseqs"] = list(np.asarray(data["dTseqs"])[idxes])
#     data["tseqs"] = list(np.asarray(data["tseqs"])[idxes])
#     data["pitchseqs"] = list(np.asarray(data["pitchseqs"])[idxes])

#     return data


# def augment(xdTs, xTs, xPs):
#     for xdT, xT, xP in zip(xdTs, xTs, xPs):
#         transpose(xP)
