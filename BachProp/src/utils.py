import midi
import copy
import numpy as np

import tqdm, mido

def filterRareRhythms(dataset, dictionaries, thres=200):
    flatten = []
    for label, score in dataset.items():
        for x in score['T']:
            flatten.append(x)
            
    hist = [(entry, flatten.count(entry)) for entry in dictionaries['T']]   
    
    retained_fraction = 0.

    while retained_fraction < 0.95:
        rare_rhythms = [x[0] for x in hist if x[1] < thres]
        filtered_dataset = {}
        for label, score in dataset.items():
            contains_rare_rhythms = False
            for x in score['T']:
                if x in rare_rhythms:
                    contains_rare_rhythms = True
                    break
            if not contains_rare_rhythms:
                filtered_dataset[label] = score

        retained_fraction = float(len(filtered_dataset))/len(dataset)
        thres = int(thres / 2)
    print("Kept %i/%i (%.2f) melodies"%(len(filtered_dataset), len(dataset), retained_fraction))
    return filtered_dataset

def mergeTrack(s):
    """
    Merge all tracks in s in a single one.
    """
    singletrack = midi.Track()
    events = []
    for i, track in enumerate(s):
        t = 0
        for event in track:
            t += event.tick
            if event.name in ['Note On', 'Note Off']:
                candidate = {'t': t, 'event': event}
                events.append(candidate)
    events = sorted(events, key=lambda k: k['t'])
    tick = 0
    for e in events:
        e['event'].tick = e['t'] - tick
        tick = e['t']
        singletrack.append(e['event'])
    return singletrack

def parseMIDI(midi_file):
    s = midi.read_midifile(midi_file)
    tpb = float(s.resolution)
    events = mergeTrack(s)
    T = []
    P = []
    dT = []
    dt = 0
    for n, event in enumerate(events):
        if event.name == 'Note On' and event.data[1] > 0:
            pitch_n = event.data[0]
            n2 = n
            duration_n = 0
            while True:
                n2 += 1
                if n2 > (len(events)-1):
                    break
                duration_n += events[n2].tick
                if events[n2].data[0] == pitch_n and events[n2].name == 'Note Off':
                    break
                if events[n2].data[0] == pitch_n and events[n2].name == 'Note On' and events[n2].data[1] == 0:
                    break
            if duration_n > 0.:
                P.append(pitch_n)
                T.append(duration_n)
                dT.append(event.tick+dt)
            dt = 0
        elif event.name == 'Note Off' or event.data[1] == 0:
            dt += event.tick

    return dT, T, P, tpb

def getDictionaries(dataset, durations=None):
    p_text = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'bB', 'B']

    dictionaries = {"T": [], "P": [], "dT": [], "P_text": [], "dT_text": [], "T_text": []}
    for key in ['dT', 'T', 'P']:
        flatten = []
        for label, score in dataset.items():
            for x in score[key]:
                flatten.append(x)

        dictionaries[key] = sorted(list(set(flatten)))
    for p in dictionaries['P']:
        dictionaries['P_text'].append(p_text[p%12]+str(p//12))
    if durations is None:
        for dt in dictionaries['dT']:
            dictionaries['dT_text'].append(str(dt))
        for t in dictionaries['T']:
            dictionaries['T_text'].append(str(t))
    else:
        for dt in dictionaries['dT']:
            dictionaries['dT_text'].append(durations['text'][durations['val'].index(dt)])
        for t in dictionaries['T']:
            dictionaries['T_text'].append(durations['text'][durations['val'].index(t)])
        
    return dictionaries


def findClosest(val, vec):
    """
    find the element closest to val in vec 
    """
    diff = [abs(float(el)-float(val)) for el in vec]
    idx = np.argmin(diff)
    return vec[idx]

def normalizeDuration(dataset):
    # Duration
    durations = {'val': [], 'text': []}
    
    durations['val'].append(0.)
    durations['text'].append('0')
    for x in range(0,8):
        #[1./4., 1./2., 1., 2., 4.]
        note = 2**x
        quarterlength = 16./note
        
        if note == 1:
            notetext = ".25"
        elif note == 2:
            notetext = ".5"
        else:
            notetext = str(int(note/4.))
        #Add the value
        durations['val'].append(quarterlength)
        durations['text'].append(notetext)
        #The triplets
        durations['val'].append(quarterlength/3.)
        durations['text'].append(notetext+")3")
        #The dotted values
        durations['val'].append(quarterlength+quarterlength/2.)
        durations['text'].append(notetext+'.')
    
    sortidxes = np.argsort(durations['val'])
    durations['val'] = list(np.asarray(durations['val'])[sortidxes])
    durations['text'] = list(np.asarray(durations['text'])[sortidxes])

    
    normalized_dataset = {}
    for label, data in dataset.items():
        normalized_dataset[label] = copy.deepcopy(data)
        normalized_dataset[label]['T'] = []
        normalized_dataset[label]['dT'] = []
        normalized_dataset[label]['P'] = []

        tpb = float(data['TPB'])
        for n in range(len(data['T'])):
            tn = findClosest(data['T'][n]/tpb, durations['val']) 
            if tn == 0:
                print("0 duration note %i, %.2f"%(data['T'][n], data['T'][n]/tpb))
                print(durations['val'])
            else:
                normalized_dataset[label]['T'].append(tn)
                normalized_dataset[label]['dT'].append(findClosest(data['dT'][n]/tpb, durations['val']))
                normalized_dataset[label]['P'].append(data['P'][n])
    
    return normalized_dataset, durations

def augment(dataset, dictionaries, offset=0):
    #-offset and +offset makes sure we explore all possible keys
    pitch_lower_bound = min(dictionaries['P'])-offset
    pitch_upper_bound = max(dictionaries['P'])+offset
    #print pitch_lower_bound, pitch_upper_bound
    augmented_dataset = {}
    for label, data in dataset.items():
        minpitch = min(data['P'])
        maxpitch = max(data['P'])
        #print minpitch, maxpitch
        possible_shifts = range(pitch_lower_bound-minpitch, pitch_upper_bound-maxpitch)
        #print possible_shifts
        for shift in possible_shifts:
            #print shift
            newlabel = label+"_transposed_%i"%(shift)
            augmented_dataset[newlabel] = copy.deepcopy(data)
            #print augmented_dataset[newlabel]['P'][:10]
            augmented_dataset[newlabel]['P'] = [x+shift for x in data['P']]
    print("Augment data: from %i to %i scores (x %i)"%(len(dataset), len(augmented_dataset), len(augmented_dataset)/len(dataset)))
    
    return augmented_dataset

def tokenize(dataset, dictionaries):
    xP = []
    xT = []
    xdT = []
    for label, melody in dataset.items():
        xP.append([dictionaries["P"].index(x) for x in melody["P"]])
        xT.append([dictionaries["T"].index(x) for x in melody["T"]])
        xdT.append([dictionaries["dT"].index(x) for x in melody["dT"]])
    return xdT, xT, xP

def writeMIDI(dtseq, Tseq, pitchseq, path, label="1", tag="1", resolution=140):

    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(format = 0, resolution = resolution)
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    tick = 0
    Events = []
    
    #Timing
    timeperbeat = 60. / 120#[s]
    timepertick = timeperbeat/resolution
    
    for dt, T, p in zip(dtseq, Tseq, pitchseq):
        if dt == 'START/END' or T == 'START/END' or p == 'START/END':
            break
        tick = tick + int(dt*resolution)
        Events.append({'t': tick, 'p': p, 'm': 'ON'})
        Events.append({'t': tick+int(T*resolution), 'p': p, 'm': 'OFF'})

    Events = sorted(Events, key=lambda k: k['t'])
    tick = 0
    for event in Events:
        if event['m'] == 'ON':
            e =  midi.NoteOnEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        if event['m'] == 'OFF':
            e =  midi.NoteOffEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        track.append(e)
        tick = event['t']
        

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    midi.write_midifile(path+label+"_"+tag+".mid", pattern)
    
    return pattern

def longMIDI(dtseqs, Tseqs, pitchseqs, path, label="1", tag="all", resolution=140):
    ended = False
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(format = 0, resolution = resolution)
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    tick = 0
    Events = []
    i = 0
    for dtseq, Tseq, pitchseq in zip(dtseqs, Tseqs, pitchseqs):
        i+=1
        last_dur = 0
        ended = False
        for dt, T, p in zip(dtseq, Tseq, pitchseq):
            if dt == 'START/END' or T == 'START/END' or p == 'START/END':
                ended = True
                tick = tick + int(resolution*last_dur) + resolution*4
                continue
            tick = tick + int(dt*resolution)
            Events.append({'t': tick, 'p': p, 'm': 'ON'})
            Events.append({'t': tick+int(T*resolution), 'p': p, 'm': 'OFF'})
            last_dur = T
        tick = tick + int(resolution*last_dur) + resolution*4
        if ended:
            writeMIDI(dtseq, Tseq, pitchseq, path=path, label=label, tag=str(i))
    Events = sorted(Events, key=lambda k: k['t'])
    tick = 0
    for event in Events:
        if event['m'] == 'ON':
            e =  midi.NoteOnEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        if event['m'] == 'OFF':
            e =  midi.NoteOffEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        track.append(e)
        tick = event['t']
        
        if tick * (60. / (120.*resolution)) > 3600.:
            ended = True
            break
        
    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    midi.write_midifile(path+label+"_"+tag+".mid", pattern)
    midifile = mido.MidiFile(path+label+"_"+tag+".mid")
    
    print(midifile.length, tick * (60. / (120.*resolution)), len(midifile.tracks), midifile.type)
    if ended:
        print("Long enough...YEY")
    return pattern

def sample(preds, temperature=1.):
    if temperature == 0.:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float32')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)
    out = np.zeros(shape=preds.shape, dtype='float32')
    out[song_idx,t,np.argmax(probas)] = 1.
    return out

def sampleNmax(preds, N=3):
    candidateidxes = np.argsort(preds)[-N:]
    prcandidates = preds[candidateidxes]
    return candidateidxes[sample(prcandidates)]


