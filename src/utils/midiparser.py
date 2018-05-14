import midi
import os, sys
import cPickle
from bisect import bisect_left
import random


def save(params, filename):
    f = file(filename, 'wb')
    cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load(filename):
    f = file(filename, 'rb')
    loaded_params = cPickle.load(f)
    f.close()
    return loaded_params


def mergeTrack(s):
    singletrack = midi.Track()
    events = []
    for i, track in enumerate(s):
        t = 0
        for event in track:
            t += event.tick
            if event.name in ['Note On', 'Note Off']:
                candidate = {'t': t, 'event': event}
                if candidate not in events:
                    events.append(candidate)
    events = sorted(events, key=lambda k: k['t'])
    tick = 0
    for e in events:
        e['event'].tick = e['t'] - tick
        tick = e['t']
        singletrack.append(e['event'])

    return singletrack


def parseMIDIfile(pathtofile, dictionaries, verbose=False):
    s = midi.read_midifile(pathtofile)
    tpb = float(s.resolution)  # ticks per beat (pulse per quarter note)
    events = mergeTrack(s)
    song = {"messageseq": [], "pitchseq": [], "tickseq": []}
    myrepr = {"dt": [], "T": [], "p": []}
    cumuldt = 0
    if verbose:
        print(s.resolution, tpb)
    for idx, event in enumerate(events):
        try:
            if verbose:
                print(event)
            if event.name == 'Note On' and event.data[1] > 0:
                pitch_t = event.data[0]

                idx2 = idx
                T = 0
                while True:
                    idx2 += 1
                    if idx2 > (len(events) - 1):
                        break
                    T += events[idx2].tick
                    if events[idx2].data[0] == pitch_t and events[idx2].name == 'Note Off':
                        break
                    if events[idx2].data[0] == pitch_t and events[idx2].name == 'Note On' and events[idx2].data[1] == 0:
                        break

                if T == 0.:  # Don't consider 0 duration notes
                    continue

                candidateT = takeClosest(T / tpb, dictionaries["duration"])
                if candidateT == 0.:  # note that are maped to 0 duration
                    cumuldt += event.tick
                    continue
                if pitch_t > max(dictionaries["pitch"]) or pitch_t < min(dictionaries["pitch"]):
                    print("pitch", pitch_t, "off bound")
                    return None

                myrepr["p"].append(event.data[0])
                myrepr["T"].append(candidateT)
                dt = event.tick + cumuldt
                myrepr["dt"].append(takeClosest(dt / tpb, dictionaries["duration"]))
                if verbose:
                    print(dt, takeClosest(dt / tpb, dictionaries["duration"]), T, candidateT, event.data[0])
                cumuldt = 0
            elif event.name == 'Note Off' or event.data[1] == 0:
                cumuldt += event.tick
        except AttributeError:
            print("Warning: maybe unrecognized event", event)
            return None
        except Exception as err:
            print("Error: ", err)
            return None

    return myrepr["dt"], myrepr["T"], myrepr["p"]


def takeClosest(myNumber, myList):
    """
	Assumes myList is sorted. Returns closest value to myNumber.

	If two numbers are equally close, return the smallest number.
	"""
    # precision = 0.2
    # myList = [x for x in dictionaries["duration"]]
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    diff1 = after - myNumber
    diff2 = myNumber - before

    if diff1 < diff2:
        # print(diff1)
        return after
    else:
        # print(diff2)
        return before


def parseFolder(datapath, dictionaries, verbose=False):
    DTs = []
    Ts = []
    Ps = []
    labels = []

    for filename in os.listdir(datapath):
        try:
            if filename[-4:] in [".MID", ".mid", ".squ"]:
                name = filename[:-4]
                if verbose:
                    print("Processing " + datapath + filename)
                res = None
                if 'all' not in name:
                    res = parseMIDIfile(datapath + filename, dictionaries)
                if res is not None:
                    dtseq, Tseq, pitchseq = res
                    DTs.append(dtseq)
                    Ts.append(Tseq)
                    Ps.append(pitchseq)
                    labels.append(name)
                elif verbose:
                    print("-->skipped")
        except Warning as err:
            print("Warning: {}".format(err))
            continue
        except KeyboardInterrupt:
            break
        except Exception as err:
            print("Error: {}".format(err))
            continue

    return {"dTseqs": DTs, "tseqs": Ts, "pitchseqs": Ps}, labels


def getDictionaries():
    dictionaries = {"pitch": [], "duration": [], "pitch_text": [], "duration_text": []}

    # A defined range of pitches
    dictionaries["pitch"] = range(0, 127)
    text = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'bB', 'B']
    for p in dictionaries["pitch"]:
        dictionaries["pitch_text"].append(text[p % 12] + str(p // 12))

    # Duration

    dictionaries["duration"].append((0., '0'))  # Instantaneous

    for x in range(0, 8):
        # [1./4., 1./2., 1., 2., 4.]
        note = 2 ** x
        quarterlength = 16. / note

        if note == 1:
            notetext = ".25"
        elif note == 2:
            notetext = ".5"
        else:
            notetext = str(note / 4)
        # Add the value
        dictionaries["duration"].append((quarterlength, notetext))
        # The triplets
        dictionaries["duration"].append((quarterlength * 1. / 3., notetext + ")3"))
        # The dotted values
        dictionaries["duration"].append((quarterlength + quarterlength / 2., notetext + '.'))

    dictionaries["duration"] = sorted(list(dictionaries["duration"]))

    for i, d in enumerate(dictionaries["duration"]):
        dictionaries["duration_text"].append(d[1])
        dictionaries["duration"][i] = d[0]

    return dictionaries


def writeMIDI(dtseq, Tseq, pitchseq, path="", label="", tag="retrieved", resolution=192):
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(format=0, resolution=resolution)
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    tick = 0
    Events = []

    for dt, T, p in zip(dtseq, Tseq, pitchseq):
        if dt == "END" or T == "END" or p == "END":
            break
        tick = tick + int(dt * resolution)
        Events.append({'t': tick, 'p': p, 'm': 'ON'})
        Events.append({'t': tick + int(T * resolution), 'p': p, 'm': 'OFF'})

    Events = sorted(Events, key=lambda k: k['t'])
    tick = 0
    for event in Events:
        if event['m'] == 'ON':
            e = midi.NoteOnEvent(tick=event['t'] - tick, velocity=90, pitch=event['p'])
        if event['m'] == 'OFF':
            e = midi.NoteOffEvent(tick=event['t'] - tick, velocity=90, pitch=event['p'])
        track.append(e)
        tick = event['t']

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    midi.write_midifile(path + label + "_" + tag + ".mid", pattern)
    return pattern


def cleanDic(data, dictionaries, clean=False, verbose=False):
    out = {"pitchseqs": [], "dTseqs": [], "tseqs": [], "duration_text": [], "pitch_text": []}
    for key, value in data.iteritems():
        flatten = [item for song in value for item in song[:-1]]
        if verbose:
            print("########--" + key + "--#########")

        if key == 'pitchseqs':
            if clean:
                minpitch = min(flatten)
                maxpitch = max(flatten)
                out[key] = range(minpitch, maxpitch + 1)
                out["pitch_text"] = dictionaries["pitch_text"][minpitch:maxpitch + 1]
            else:
                minpitch = min(flatten)
                maxpitch = max(flatten)
                out[key] = range(minpitch, maxpitch + 1)
                out["pitch_text"] = dictionaries["pitch_text"][minpitch:maxpitch + 1]

            # out['pitchseqs'] = dictionaries["pitch"]
            # out["pitch_text"] = dictionaries["pitch_text"]
            for x, t in zip(out[key], out["pitch_text"]):
                count = flatten.count(x)
                if verbose:
                    print(x, t, count)
        else:
            for x, t in zip(dictionaries["duration"], dictionaries["duration_text"]):
                count = flatten.count(x)
                if count > 10 and clean:
                    out[key].append(x)
                    if t not in out["duration_text"]:
                        out["duration_text"].append(t)
                if verbose:
                    print(x, t, count)
            if not clean:
                out[key] = dictionaries["duration"]

        if verbose:
            print("Dim: ", len(out[key]))

    if not clean:
        out["duration_text"] = dictionaries["duration_text"]

    return out


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasetname = sys.argv[1]
    else:
        datasetname = "JSB_Chorales"

    datapath = "../../data/" + datasetname + '/'
    print("Processing " + datasetname)

# outpath = "../out/"
# if not os.path.exists(outpath):
# 	os.mkdir(outpath)

# outpath = "../out/"+datasetname+'/'
# if not os.path.exists(outpath):
# 	os.mkdir(outpath)

# midipath = "../out/"+datasetname+'/midi/'
# if not os.path.exists(midipath):
# 	os.mkdir(midipath)

# dictionaries = getDictionaries()

# dataset = parseFolder(datapath, dictionaries)

# dictionaries = cleanDic(dataset)

# print(dictionaries)
# print(len(dataset['pitchseqs']), "saved songs")

# save([dataset, dictionaries], outpath+datasetname+".pkl")
