import utils
import numpy as np
import pickle
import tqdm
import os, sys
from sys import stdout

import keras
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, load_model
from keras.layers import Input, Masking, TimeDistributed, Dense, Concatenate, Dropout, LSTM, GRU
from keras.optimizers import Adam

class BachProp:
    """
    Class defining BachProp
    """

    def __init__(self, chorpus, TBPTT_size=3*100, batch_size=32):
        self.datapath = "../data/"
        self.loadmodelpath = "../load/BachProp/"
        self.chorpus = chorpus
        self.outpath = "../save/BachProp/"+chorpus+'/'
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.genpath = self.outpath + "midi/"
        if not os.path.exists(self.genpath):
            os.makedirs(self.genpath)
        

        self.TBPTT_size = TBPTT_size
        self.batch_size = batch_size


        self.IO = None
        self.dataset = None

        self.log = {'loss': [], 'val_loss': [], 'dT_acc': [], 'val_dT_acc': [], 'T_acc': [], 'val_T_acc': [], 'P_acc': [], 'val_P_acc': [], }
        self.best_val_accP = 0.


    def processData(self):
        dataset = {}
        for filename in os.listdir(self.datapath+self.chorpus+"/midi/"):
            if filename[-3:] in ["mid", "MID", "SQU", "KPP", "squ", "kpp"]:
                label = filename[:-4]
                try:
                    dTseq, Tseq, Pseq, tpb = utils.parseMIDI(self.datapath+self.chorpus+"/midi/"+filename)
                except TypeError:
                    continue
                dataset[label] = {}
                dataset[label]['T']= Tseq
                dataset[label]['dT']= dTseq
                dataset[label]['P']= Pseq
                dataset[label]['TPB'] = tpb

        dictionaries = utils.getDictionaries(dataset)

        filtered_dataset = utils.filterRareRhythms(dataset, dictionaries)
        filtered_dictionaries = utils.getDictionaries(filtered_dataset)

        normalized_dataset, durations = utils.normalizeDuration(filtered_dataset)
        normalized_dictionaries = utils.getDictionaries(normalized_dataset, durations)

        augmented_dataset = utils.augment(normalized_dataset, normalized_dictionaries)
        augmented_dictionaries = utils.getDictionaries(augmented_dataset, durations)

        self.dataset = augmented_dataset
        self.dictionaries = augmented_dictionaries

        print(self.dictionaries)

    def saveData(self):
        pickle.dump([self.dataset, self.dictionaries], open(self.datapath+self.chorpus+"/data.pkl", "wb" ))

    def loadData(self):
        if os.path.exists(self.datapath+self.chorpus+"/data.pkl"):
            self.dataset, self.dictionaries = pickle.load( open(self.datapath+self.chorpus+"/data.pkl", "rb" ) )
        else:
            print("No data for %s has been found in the %s repository, processing %s instead"%(self.chorpus, self.datapath, self.datapath+self.chorpus))
            self.processData()
            self.saveData()
        
    def checkRepresentation(self):
        label = str(np.random.choice(list(self.dataset.keys()), 1)[0])
        utils.writeMIDI(self.dataset[label]['dT'][1:-1], self.dataset[label]['T'][1:-1], self.dataset[label]['P'][1:-1], path="../data/"+self.chorpus+"/", label=label, tag='retrieved')
        # selection = {}
        # selection['labels'] =  np.random.choice(list(self.dataset.keys()), 1)
        # selection['idxes'] = [list(self.dataset.keys()).index(l) for l in selection['labels']]
        # dTs, Ts, Ps = self.ANN2data(self.IO['XdT'][selection['idxes']], self.IO['XT'][selection['idxes']], self.IO['XP'][selection['idxes']])
        # utils.longMIDI(dTs, Ts, Ps, path="../data/"+self.chorpus+"/", tag='retrieved')
        # print(selection['labels'])



    def ANN2data(self, XdTs, XTs, XPs):
        """
        Translate back from one-hot matrices to note sequences
        """
        dTs = []
        Ts = []
        Ps = []

        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)

        for XdT, XT, XP in zip(XdTs[:,1:], XTs[:,1:], XPs[:,1:]):
            xdT = np.where(XdT == 1)[1]
            xT = np.where(XT == 1)[1]
            xP = np.where(XP == 1)[1]
            dT = []
            T = []
            P = []
            for dt, t, p in zip(xdT, xT, xP):
                dt = self.dictionaries['dT'][dt]
                t = self.dictionaries['T'][t]
                p = self.dictionaries['P'][p]

                dT.append(dt)
                T.append(t)
                P.append(p)

            dTs.append(dT)
            Ts.append(T)
            Ps.append(P)
            
        return dTs, Ts, Ps


    def data2ANN(self):

        if self.dataset is None:
            self.loadData()


        #Add start/end tags
        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)
        for label, score in self.dataset.items():
            if tag not in score['dT']:
                score['dT'].insert(0,tag)
                score['T'].insert(0,tag)
                score['P'].insert(0,tag)
                score['dT'].append(tag)
                score['T'].append(tag)
                score['P'].append(tag)


        xdT, xT, xP = utils.tokenize(self.dataset, self.dictionaries)

        P = [np_utils.to_categorical(x, len(self.dictionaries['P'])) for x in xP]
        T = [np_utils.to_categorical(x, len(self.dictionaries['T'])) for x in xT]
        dT = [np_utils.to_categorical(x, len(self.dictionaries['dT'])) for x in xdT]

        seqlens = [len(X) for X in P]
        len95Perc = int(np.mean(seqlens)+2*np.std(seqlens))

        maxlen = len95Perc + int(self.TBPTT_size/3-len95Perc%(self.TBPTT_size/3))

        dT = pad_sequences(dT, value=0., dtype="int32", padding="post", truncating="post", maxlen=maxlen)
        T = pad_sequences(T, value=0., dtype="int32", padding="post", truncating="post", maxlen=maxlen)
        P = pad_sequences(P, value=0., dtype="int32", padding="post", truncating="post", maxlen=maxlen)
        print("Longest melody length [note]: %i, mean: %i, std: %i, selected length: %i"%(max(seqlens),np.mean(seqlens),np.std(seqlens), maxlen))

        XdTall = [np.repeat(x, 3, axis=0) for x in dT]    
        XdT = np.asarray([x[2:-3] for x in XdTall], dtype=int)
        YdT = np.asarray([x[5:] for x in XdTall], dtype=int)

        XTall = [np.repeat(x, 3, axis=0) for x in T]    
        XT = np.asarray([x[1:-4] for x in XTall], dtype=int)
        YT = np.asarray([x[4:-1] for x in XTall], dtype=int)

        XPall = [np.repeat(x, 3, axis=0) for x in P]    
        XP = np.asarray([x[:-5] for x in XPall], dtype=int)
        YP = np.asarray([x[3:-2] for x in XPall], dtype=int)

        T = XdT.shape[1]

        XP = pad_sequences(XP, value=0., dtype="int32", padding="post", maxlen = T + (self.TBPTT_size-T%self.TBPTT_size))
        YP = pad_sequences(YP, value=0., dtype="int32", padding="post", maxlen = T + (self.TBPTT_size-T%self.TBPTT_size))
        XT = pad_sequences(XT, value=0., dtype="int32", padding="post", maxlen = T + (self.TBPTT_size-T%self.TBPTT_size))
        YT = pad_sequences(YT, value=0., dtype="int32", padding="post", maxlen = T + (self.TBPTT_size-T%self.TBPTT_size))
        XdT = pad_sequences(XdT, value=0., dtype="int32", padding="post", maxlen = T + (self.TBPTT_size-T%self.TBPTT_size))
        YdT = pad_sequences(YdT, value=0., dtype="int32", padding="post", maxlen = T + (self.TBPTT_size-T%self.TBPTT_size))

        #self.TBPTT_step_number = XdT.shape[1]//self.TBPTT_size

        print("Final I/O data shape:")
        print('dT', XdT.shape)
        print('T', XT.shape)
        print('P', XP.shape)


        self.TBPTT_steps = []
        for X in XP:
            steps = 0
            while np.sum(X[steps*self.TBPTT_size:(steps+1)*self.TBPTT_size]) > 0:
                steps += 1
            self.TBPTT_steps.append(steps)
        self.TBPTT_steps = np.asarray(self.TBPTT_steps)

        IO = {'XP': XP, 'YP': YP, 'XT': XT, 'YT': YT, 'XdT': XdT, 'YdT': YdT, 'TBTT_steps': self.TBPTT_steps}

        self.IO = IO


    def buildModel(self, dropout=0.1, recurrent_dropout=0.):
        X = dict()
        M = dict()
        H = dict()
        Y = dict()
        
        dTvocsize = len(self.dictionaries['dT'])
        Tvocsize = len(self.dictionaries['T'])
        Pvocsize = len(self.dictionaries['P'])
        
        X['dT'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, dTvocsize), name='XdT')
        X['T'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, Tvocsize), name='XT') 
        X['P'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, Pvocsize), name='XP')

        M['dT'] = Masking(mask_value=0.)(X['dT'])
        M['T'] = Masking(mask_value=0.)(X['T'])
        M['P'] = Masking(mask_value=0.)(X['P'])

        H['dT1'] = GRU(16, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(M['dT'])

        H['T1'] = GRU(64, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(M['T'])

        H['P1'] = GRU(128, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(M['P'])

        H['common'] = Concatenate()([H['dT1'], H['T1'], H['P1']])
        H['common'] = GRU(256, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(H['common'])

        H['dT2'] = GRU(16, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(H['common'])

        H['T2'] = GRU(64, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(H['common'])

        H['P2'] = GRU(128, 
            return_sequences=True, 
            stateful=True, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
            )(H['common'])

        Y['dT'] = Concatenate()([H['dT1'], H['dT2']])
        Y['dT'] = TimeDistributed(Dense(dTvocsize, activation='softmax'), name='YdT')(Y['dT'])
        Y['T'] = Concatenate()([H['T1'], H['T2']])
        Y['T'] = TimeDistributed(Dense(Tvocsize, activation='softmax'), name='YT')(Y['T'])
        Y['P'] = Concatenate()([H['P1'], H['P2'], H['common']])
        Y['P'] = TimeDistributed(Dense(Pvocsize, activation='softmax'), name='YP')(Y['P'])

        self.model = Model(inputs = [X['dT'], X['T'], X['P']], outputs = [Y['dT'], Y['T'], Y['P']])
        #opt = Adam(clipnorm=5.) 
        opt = Adam() 
        self.model.compile(
            loss='categorical_crossentropy', 
            optimizer=opt,
            metrics=['acc'],
            loss_weights=[.1, .3, .6])
        self.model.summary()

    def loadModel(self):
        if os.path.exists(self.loadmodelpath+self.chorpus+'.model'):
            self.model = load_model(self.loadmodelpath+self.chorpus+".model")
            self.log = pickle.load(open(self.loadmodelpath+self.chorpus+".log", "rb" ))
            self.best_val_accP = max(self.log['val_P_acc'])
        else:
            print("No model for %s has been found in the %s repository, building a new one instead"%(self.chorpus, self.loadmodelpath))
            self.buildModel()
            self.log = {'loss': [], 'val_loss': [], 'dT_acc': [], 'val_dT_acc': [], 'T_acc': [], 'val_T_acc': [], 'P_acc': [], 'val_P_acc': [], }
    
    def saveModel(self):
        self.model.save(self.outpath+self.chorpus+".model")
        pickle.dump(self.log, open(self.outpath+self.chorpus+".log", "wb" ))

    def trainModel(self, epochs=100, validation_split=0.2):

        if self.IO is None:
            self.data2ANN()

        print('\nTraining Model')

        nb_samples = self.IO['XP'].shape[0]
        all_idxes = np.asarray(range(nb_samples))
        all_batch_idxes = {'train': {}, 'valid': {}}

        for step in set(self.TBPTT_steps):
            idxes = all_idxes[np.where(self.TBPTT_steps==step)]
            idxes = list(idxes)
            nb_samples = len(idxes)
            split_idx = int((1 - validation_split) * nb_samples)
            #split_idx -= split_idx % self.batch_size
            all_batch_idxes['train'][str(step)] = idxes[:split_idx]
            all_batch_idxes['train'][str(step)].extend(
                                        np.random.choice(idxes[:split_idx], 
                                                self.batch_size - len(all_batch_idxes['train'][str(step)]) % self.batch_size))
            all_batch_idxes['valid'][str(step)] = idxes[split_idx:]
            all_batch_idxes['valid'][str(step)].extend(
                                        np.random.choice(idxes[split_idx:], 
                                                self.batch_size - len(all_batch_idxes['valid'][str(step)]) % self.batch_size))

        self.TBPTT_steps = set(self.TBPTT_steps)

        for epoch in range(epochs):
            tr_epoch_res = []
            for step_number in self.TBPTT_steps:
                #Reshape for batches
                np.random.shuffle(all_batch_idxes['train'][str(step_number)])
                batch_idxes = np.reshape(all_batch_idxes['train'][str(step_number)], (-1,self.batch_size))
                batch_nbr_per_epoch = len(batch_idxes)
                for batch, idxes in enumerate(batch_idxes):
                    self.model.reset_states()
                    batch_res = []
                    for step in range(step_number):
                        res = self.model.train_on_batch({'XdT': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YdT': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                   'YT': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                   'YP': self.IO['YP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        tr_epoch_res.append(res)
            tr_epoch_res = np.mean(tr_epoch_res, axis=0)
            loss, lossdT, lossT, lossP, accdT, accT, accP = tr_epoch_res
            print("Epoch %i/%i - \tloss: %.2f - dT: %.2f - T: %.2f - P: %.2f" %
                 (epoch+1,epochs,loss,accdT, accT, accP))
            
            val_epoch_res = []
            for step_number in self.TBPTT_steps:
                #Reshape for batches
                np.random.shuffle(all_batch_idxes['valid'][str(step_number)])
                batch_idxes = np.reshape(all_batch_idxes['valid'][str(step_number)], (-1,self.batch_size))
                batch_nbr_per_epoch = len(batch_idxes)
                for batch, idxes in enumerate(batch_idxes):
                    self.model.reset_states()
                    batch_res = []
                    for step in range(step_number):
                        res = self.model.test_on_batch({'XdT': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YdT': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                   'YT': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                   'YP': self.IO['YP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        val_epoch_res.append(res)
            val_epoch_res = np.mean(val_epoch_res, axis=0)
            val_loss, val_lossdT, val_lossT, val_lossP, val_accdT, val_accT, val_accP = val_epoch_res
            print("\t\tloss: %.2f - dT: %.2f - T: %.2f - P: %.2f" %
                 (val_loss,val_accdT, val_accT, val_accP))
            
            self.log['loss'].append(loss)
            self.log['dT_acc'].append(accdT)
            self.log['T_acc'].append(accT)
            self.log['P_acc'].append(accP)
            self.log['val_loss'].append(val_loss)
            self.log['val_dT_acc'].append(val_accdT)
            self.log['val_T_acc'].append(val_accT)
            self.log['val_P_acc'].append(val_accP)
            
            if val_accP > self.best_val_accP:
                self.best_val_accP = val_accP
                self.saveModel()

    def generate(self, note_len=200, until_all_ended=True):


        if until_all_ended:
            note_len = 100000
        
        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)

        dTvocsize = len(self.dictionaries['dT'])
        Tvocsize = len(self.dictionaries['T'])
        Pvocsize = len(self.dictionaries['P'])
        n_ex = self.batch_size

        ended = [False]*n_ex

        XdTs_hat = np.zeros((n_ex, note_len+1, dTvocsize), dtype = int)
        XdTs_hat[:,0,-1] = 1
        XdTs_probs = np.zeros((n_ex, note_len+1, dTvocsize), dtype = float)

        XTs_hat = np.zeros((n_ex, note_len+1, Tvocsize), dtype = int)
        XTs_hat[:,0,-1] = 1
        XTs_probs = np.zeros((n_ex, note_len+1, Tvocsize), dtype = float)

        XPs_hat = np.zeros((n_ex, note_len+1, Pvocsize), dtype = int)
        XPs_hat[:,0,-1] = 1
        XPs_probs = np.zeros((n_ex, note_len+1, Pvocsize), dtype = float)
        
        xdt_t = np.zeros((n_ex, self.TBPTT_size, dTvocsize), dtype=int)
        xdt_tp1 = np.zeros((n_ex, self.TBPTT_size, dTvocsize), dtype=int)
        xt_t = np.zeros((n_ex, self.TBPTT_size, Tvocsize), dtype=int)
        xt_tp1 = np.zeros((n_ex, self.TBPTT_size, Tvocsize), dtype=int)
        xp_t = np.zeros((n_ex, self.TBPTT_size, Pvocsize), dtype=int)
        
        self.model.reset_states()
        

        try:
            last_note = 0
            for t in range(note_len):  
                xdt_t[:,0] = XdTs_hat[:,t]
                xt_t[:,0] = XTs_hat[:,t]
                xp_t[:,0] = XPs_hat[:,t]
                
                probs, _, _ = self.model.predict([xdt_t, xt_t, xp_t])
                XdTs_probs[:,t] = probs[:, 0]  
                for idx in range(n_ex):  
                    dT_np1 = utils.sampleNmax(probs[idx, 0])
                    XdTs_hat[idx, t+1, dT_np1] = 1
                    if self.dictionaries['dT'][dT_np1] == 'START/END':
                        if ended[idx] == False:
                            print("%i dT ended @note #%i"%(idx+1, t))
                        ended[idx] = True 
                        
                xdt_tp1[:,0] = XdTs_hat[:,t+1]

                _, probs, _ = self.model.predict([xdt_tp1, xt_t, xp_t])
                XTs_probs[:,t] = probs[:, 0]          
                for idx in range(n_ex):  
                    T_np1 = utils.sampleNmax(probs[idx, 0])
                    XTs_hat[idx, t+1, T_np1] = 1
                    if self.dictionaries['T'][T_np1] == 'START/END':
                        if ended[idx] == False:
                            print("%i T ended @note #%i"%(idx+1, t))
                        ended[idx] = True 
                xt_tp1[:,0] = XTs_hat[:,t+1]

                _, _, probs = self.model.predict([xdt_tp1, xt_tp1, xp_t])
                XPs_probs[:,t] = probs[:, 0]
                for idx in range(n_ex):  
                    P_np1 = utils.sampleNmax(probs[idx, 0])
                    XPs_hat[idx, t+1, P_np1] = 1.     
                    if self.dictionaries['P'][P_np1] == 'START/END':
                        if ended[idx] == False:
                            print("%i P ended @note #%i"%(idx+1, t))
                        ended[idx] = True 
                
                last_note = t
                if until_all_ended == True and np.sum(ended) == n_ex:
                    break
        except KeyboardInterrupt:
            print("ctrl-c ended @note %i"%(t))
            pass
        print("End generating: %i/%i song ended"%(np.sum(ended),n_ex))
        return [XdTs_hat[:,:last_note], XTs_hat[:,:last_note], XPs_hat[:,:last_note], XdTs_probs[:,:last_note], XTs_probs[:,:last_note], XPs_probs[:,:last_note]]

if __name__ == "__main__":

    m = BachProp("JSB_Chorales")
    m = BachProp(sys.argv[1])
    m.loadData()
    m.loadModel()

    #Train only
    m.data2ANN()
    m.trainModel(epochs=200)

    #Generate up to 32 complete songs
    XdTs_hat, XTs_hat, XPs_hat, XdTs_probs, XTs_probs, XPs_probs = m.generate()
    dTs, Ts, Ps = m.ANN2data(XdTs_hat, XTs_hat, XPs_hat)
    longMIDI(dTs, Ts, Ps, path=m.genpath, label=str(np.random.randint(100)))




