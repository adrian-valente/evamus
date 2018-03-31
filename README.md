Evaluation toolbox for generative models of music.
With Florian Colombo, at EPFL (Laboratory of Computational Neuroscience).

# Usage
The current steps of usage of the toolbox can be seen in the Jupyter notebook Evaluation-pipeline.ipynb

The first step is to load a dataset and preprocess with the following lines:
```
from utils.preprocessing import preprocess
dataset, sizes, dictionaries = preprocess(datapath)
```
Where `datapath` indicates a directory containing MIDI files. The files are read, simplified (single track, limited number of rhythms)  and converted to a standard dataset format described below.
The function `split` can also be useful to do cross-validation.

The second step is to implement a model. A model is a class that must implement the following methods:
`train(self, dataset)`: takes as input a standard preprocessed dataset. Besides training the model also calls the function `predict` (see below) on the training set and returns its result.
`predict(self, dataset)`: takes as input a standard preprocessed dataset and returns a "prediction dataset", ie. a dataset where each value is replaced by its probability distribution (as a defaultdict) predicted by the model. This result can then be fed to the metrics defined in evaluation.py.
`generate(self, n_songs=20, N=200, write_MIDI=False, dictionaries=None, path='../data/generated/')`: Generates songs based on the internal parameters, and convert them to MIDI if asked. 

Note that so far, 2 variants of Ngram models are implemented: INgram (for Independent Ngram, trains separate ngrams on the 3 feature sequences dT, t, and pitch), and FUNgram (for Feature Unrolled Ngram, trains an Ngram on a sequence where dT, t, and pitch are unrolled)

The third step is to define the set of models that one wants to evaluate along with their arguments in a dictionary, for example:
```
modelsDic = {'ingram': (INgram, {'order':1}),
            'fungram': (FUNgram, {'order': 3, 'sizes': sizes})}
```
Here, the keys of the dictionary are names for the models, and they map to a tuple whose first element is the class describing the model, and the second element a keyword arguments dictionary for the __init__ method of the model.
Then one can simply call:
```
models, metrics = compare_models(splitted, modelsDic)
```
which returns 2 dictionaries (for the trained models, and for the computed metrics). The metrics can then be visualized by calling `plot_metric`.

## Description of the standard dataset format
Here, a song is formally defined as a sequence of notes, each note being described by 3 values: dT (the time difference with the previous note), t (the duration of the note), and pitch. However, in the dataset as we define it, each song is splitted into 3 sequences (for each feature).
 
In practice, a dataset contains 3 layers:
- First it is a dictionary containing 3 keys: "dTseqs", "tseqs", "pitchseqs".
- Each key maps to a sequence of songs
- Each song being a sequence of values (floats) 
Note how each song is dispersed into the 3 buckets of the dictionary: for instance, dataset["dTseqs"][0], dataset["tseqs"][0] and dataset["pitchseqs"][0] are sequences having the same length and representing the 3 features in the same song.
