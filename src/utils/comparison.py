from midiparser import parseFolder, getDictionaries, cleanDic
from evaluation import preanalysis_chords, preanalysis_intervals, analyze_chords, analyze_intervals
from novelty import autonovelty, novelty_analysis

def reduce_dataset(dataset, dictionaries):
    out = dict()
    for label in dataset:
        out[label] = []
        for song in dataset[label]:
            out[label].append( [dictionaries[label].index(x) for x in song] )
    return out


def analyse_and_compare(dataset, ref_dataset, name, autonovelty_ref, chords_distr_ref, 
                        intervals_distr_ref, datapath, dictionaries, motifs=(2,4,8,16,32)):
    print("Dataset {}: {} songs".format(name, len(dataset["dTseqs"])))
    tv_chords = analyze_chords(None, dataset, title_prefix="Chords comparison for model "+name, 
                               real_dis=chords_distr_ref, plot=True)
    print("Total Variation Distance for chords distribution: {}".format(tv_chords))
    tv_intervals = analyze_intervals(None, dataset, title="Intervals comparison for model "+name,
                                     real_dis=intervals_distr_ref, plot=True)
    print("Total Variation Distance for intervals distribution: {}".format(tv_intervals))
    novelties = novelty_analysis(ref_dataset, dataset, motifs, autonovelty_ref,
                                 corpus2Name=name, dictionaries=dictionaries, plot=True)
    for i,motif in enumerate(motifs):
        print("Mean novelty at size {}: {}".format(motif, novelties[:,i].mean()))


def comparison(ref_dataset_path, dataset_paths, dataset_names, motif_sizes=(2, 4, 8, 16, 32)):
    """
    :param ref_dataset_path:
    :param datasets_paths: list of paths

    """

    ## PREPROCESSING OF MIDI FOLDERS ##
    print("Loading data...")
    dictionaries = getDictionaries()

    ref_dataset = parseFolder(ref_dataset_path, dictionaries)
    datasets = []
    for path in dataset_paths:
        datasets.append(parseFolder(path, dictionaries))
    
    # reduction of dictionary size  (we have to ensure that all notes from all datasets
    # are included in the dictionaries)
    data_all = {"dTseqs": list(ref_dataset["dTseqs"]), "tseqs": list(ref_dataset['tseqs']), 
                "pitchseqs": list(ref_dataset["pitchseqs"])}
    for data in datasets:
        for label in data:
            data_all[label] += data[label]
    dictionaries = cleanDic(data_all, dictionaries)  # get reduced dictionaries
    ref_dataset = reduce_dataset(ref_dataset, dictionaries)
    for i, data in enumerate(datasets):
        datasets[i] = reduce_dataset(data, dictionaries)
    print("Done.")

    ## PRE-ANALYSIS OF REFERENCE DATASET ##
    print("Preanalysis of reference dataset...")
    autonovelty_ref = autonovelty(ref_dataset, motif_sizes)
    chords_distr_ref = preanalysis_chords(ref_dataset)
    intervals_distr_ref = preanalysis_intervals(ref_dataset)
    print("Done.")

    ## ANALYSIS OF EACH DATASET ##
    for i, data in enumerate(datasets):
        print("Analysis of dataset {}".format(dataset_names[i]))
        analyse_and_compare(data, ref_dataset, dataset_names[i], autonovelty_ref, chords_distr_ref, 
                            intervals_distr_ref, dataset_paths[i], dictionaries)







