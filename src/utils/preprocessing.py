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
