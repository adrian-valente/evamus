from midiparser import parseFolder, getDictionaries, cleanDic
from evaluation import preanalysis_chords, preanalysis_intervals, analyze_chords, analyze_intervals, \
    analyze_transitions, plot_lengths, plot_distributions
from novelty import autonovelty, novelty_analysis, plot_novelties
import matplotlib as mpl
from keyAnalysis import key_analysis


def reduce_dataset(dataset, dictionaries):
    out = dict()
    for label in dataset:
        out[label] = []
        for song in dataset[label]:
            out[label].append( [dictionaries[label].index(x) for x in song] )
    return out


def analyse_and_compare(dataset, ref_dataset, name, autonovelty_ref, chords_distr_ref, 
                        intervals_distr_ref, sizes, dictionaries, labels, ref_labels, motifs=(2,3,4,5,6,8),
                        report=None, report_path=None):
    if report is None:
        print("Dataset {}: {} songs".format(name, len(dataset["dTseqs"])))
        tv_chords, _ = analyze_chords(None, dataset, title="Chords decomposition for model "+name,
                                      real_dis=chords_distr_ref, show_plot=True)
        print("Total Variation Distance for chords distribution: {}".format(tv_chords))
        tv_intervals, _ = analyze_intervals(None, dataset, title="Intervals decomposition for model "+name,
                                            real_dis=intervals_distr_ref, show_plot=True)
        print("Total Variation Distance for intervals distribution: {}".format(tv_intervals))
        novelties, _ = novelty_analysis(ref_dataset, dataset, motifs, autonovelty_ref, labels=labels,
                                        ref_labels=ref_labels, corpus2Name=name, dictionaries=dictionaries,
                                        show_plot=True)
        for i, motif in enumerate(motifs):
            print("Mean novelty at size {}: {}".format(motif, novelties[:,i].mean()))

    else:
        report.write("Basic numbers\n----------------\n\n")
        report.write("* {} songs\n".format(len(dataset["dTseqs"])))
        plot_lengths(dataset, dictionaries, name+" song lengths", plot_fp=report_path+name+"_lengths.png")
        tv_chords = analyze_chords(None, dataset, title="Chords comparison for model "+name, 
                                   real_dis=chords_distr_ref, 
                                   plot_fp=report_path+"chords_"+name+".png")
        
        report.write("* Total Variation Distance for chords distribution: {}\n".format(tv_chords))

        tv_intervals = analyze_intervals(None, dataset, title="Intervals comparison for model "+name,
                                         real_dis=intervals_distr_ref, 
                                         plot_fp=report_path+"intervals_"+name+".png")

        report.write("* Total Variation Distance for intervals distribution: {}\n".format(tv_intervals))

        analyze_transitions(dataset, sizes, dictionaries, name+' - ', plot_fp=report_path+name)

        novelties = novelty_analysis(ref_dataset, dataset, motifs, autonovelty_ref, labels=labels,
                                     ref_labels=ref_labels, corpus2Name=name, dictionaries=dictionaries,
                                     plot_fp=report_path+"novelty_"+name+".png", report=report)
        for i,motif in enumerate(motifs):
            report.write("* Mean novelty at size {}: {}\n".format(motif, novelties[:,i].mean()))

        report.write("Graphs\n-----------\n\n")
        report.write("![]("+name+"_lengths.png\n")
        report.write("![](chords_"+name+".png)\n")
        report.write("![](intervals_"+name+".png)\n")
        report.write("![](novelty_"+name+".png)\n")
        report.write("![]("+name+"_p.png)\n\n")
        report.write("![]("+name+"_T.png)\n\n")
        report.write("![]("+name+"_dT.png)\n\n")
        report.write("\n\n")
    return novelties


def comparison(ref_dataset_path, dataset_paths, dataset_names, motif_sizes=(2, 3, 4, 5, 6, 8),
               write_report=True, report_path="report/"):
    """
    :param ref_dataset_path:
    :param dataset_paths: list of paths to generated datasets
    :param dataset_names: list of corresponding model names
    :param motif_sizes: (tuple of ints) the size of motifs for novelty analysis
    :param write_report: True to write an html report (otherwise, everything is output to stdout)
    :param report_path: if write_report=True, path to generated report and graphs
    """
    mpl.rcParams['image.cmap'] = 'binary'

    # PREPROCESSING OF MIDI FOLDERS
    print("Loading data...")
    dictionaries = getDictionaries()

    ref_dataset, ref_labels = parseFolder(ref_dataset_path, dictionaries)
    datasets = []
    labels = []
    for path in dataset_paths:
        data, lbl = parseFolder(path, dictionaries)
        datasets.append(data)
        labels.append(lbl)

    # reduction of dictionary size  (we have to ensure that all notes from all datasets
    # are included in the dictionaries)
    data_all = {"dTseqs": list(ref_dataset["dTseqs"]), "tseqs": list(ref_dataset['tseqs']), 
                "pitchseqs": list(ref_dataset["pitchseqs"])}
    for data in datasets:
        for label in data:
            data_all[label] += data[label]
    dictionaries = cleanDic(data_all, dictionaries)  # get reduced dictionaries
    sizes = (len(dictionaries['dTseqs']), len(dictionaries['tseqs']), len(dictionaries['pitchseqs']))
    ref_dataset = reduce_dataset(ref_dataset, dictionaries)
    for i, data in enumerate(datasets):
        datasets[i] = reduce_dataset(data, dictionaries)
    print("Done.")

    report = None
    if write_report:
        report = open(report_path+"report.html", 'w')
        report.write("**Comparisons report**\n\n")

    # PRE-ANALYSIS OF REFERENCE DATASET
    print("Preanalysis of reference dataset...")
    autonovelty_ref = autonovelty(ref_dataset, motif_sizes)
    if write_report:
        report.write("Reference dataset\n=================\n\n")
        report.write("{} songs\n\n".format(len(ref_dataset["dTseqs"])))
    chords_distr_ref = preanalysis_chords(ref_dataset, make_plot=write_report, plot_fp=report_path+"chords-real.png")
    intervals_distr_ref = preanalysis_intervals(ref_dataset, make_plot=write_report,
                                                plot_fp=report_path+"intervals-real.png")
    analyze_transitions(ref_dataset, sizes, dictionaries, "Reference data - ", plot_fp=report_path+'ref')
    #key_analysis(ref_dataset, dictionaries, report=report)
    plot_lengths(ref_dataset, dictionaries, title="Reference dataset song lengths",
                 plot_fp=report_path+'ref-lengths.png')
    if write_report:
        report.write("Graphs\n------------\n\n")
        report.write("![](ref-lengths.png\n\n")
        report.write("![](chords-real.png)\n\n")
        report.write("![](intervals-real.png)\n\n")
        report.write("![](ref_p.png)\n\n")
        report.write("![](ref_T.png)\n\n")
        report.write("![](ref_dT.png)\n\n")
    print("Done.")

    # ANALYSIS OF EACH DATASET
    novelties = []
    chord_distributions = []
    interval_distributions = []
    for i, data in enumerate(datasets):
        print("Analysis of dataset {}".format(dataset_names[i]))
        if write_report:
            report.write("\n\n\n")
            report.write("Analysis of dataset {}\n=============================\n\n".format(dataset_names[i]))
        novelties.append(analyse_and_compare(data, ref_dataset, dataset_names[i], autonovelty_ref, chords_distr_ref,
                                             intervals_distr_ref, sizes, dictionaries, labels[i], ref_labels,
                                             motifs=motif_sizes, report=report, report_path=report_path))
        #key_analysis(data, dictionaries, report=report)
        chord_distributions.append(preanalysis_chords(data))
        interval_distributions.append(preanalysis_intervals(data))


    if write_report:
        plot_novelties(autonovelty_ref, novelties, dataset_names, motif_sizes, plot_fp=report_path+"novelties")
        plot_distributions(chords_distr_ref, chord_distributions, dataset_names, plot_fp=report_path+"chord_distrs.png")
        plot_distributions(intervals_distr_ref, interval_distributions, dataset_names, plot_fp=report_path+"intervals_distrs.png")
        report.write("Summary plots\n==========================\n\n")
        report.write("Novelty\n------------\n\n")
        report.write("![](novelties_violin.png)\n\n")
        report.write("![](novelties_box.png)\n\n")
        report.write("![](novelties_point.png)\n\n")
        report.write("Chord distributions\n------------\n\n")
        report.write("![](chord_distrs.png)\n\n")
        report.write("Interval distributions\n------------\n\n")
        report.write("![](intervals_distrs.png)\n\n")

    if write_report:
        report.write(r'<!-- Markdeep: --><style class="fallback">body{visibility:hidden;white-space:pre;font-family:monospace}</style><script src="../markdeep/markdeep.min.js"></script><script>window.alreadyProcessedMarkdeep||(document.body.style.visibility="visible")</script>')
    report.close()


if __name__=="__main__":
    comparison('../../data/JSB_Chorales/', ['../../data/JSB_Chorales/gen/fungram12/', '/Users/toroloco/Documents/etudes/info/projets/evamus/BachProp/save/BachProp/JSB_Chorales/midi/'],
               ['fungram12', 'BachProp'], report_path='../report/')
