import numpy as np
import matplotlib.pyplot as plt

def split(dataset, k):
    """
    :param dataset: a dictionary of list of sequences containing the lists 'dtseqs', 'Tseqs', 'pitchseq'
    :param k: the number of folds
    :returns: a list of dictionaries {'train':, 'test':} where train and test have the same format as dataset
    (note: train and test are views, and not copies of the original dataset)
    """
    n = len(dataset['dtseqs'])
    ret = []
    for i in xrange(k):
        ret.append({'train':{}, 'test':{}})
        for key in dataset:
            ret[-1]['train'][key] =  dataset[key][:i*(n//k)] + dataset[key][min((i+1)*(n//k), n):]
            ret[-1]['test'][key] = dataset[key][i*(n//k):min((i+1)*(n//k), n)]
    return ret

def evaluate(generated_data, reference_data):
    print "evaluated not yet defined"
    return None

def plot_metric(values, title = ""):
    """
    Plots a bar chart comparing the value of metric on different models with error bars
    :param values: a dictionary mapping model name -> a sequence of values of the metric
    """
    x_i = np.arange(len(values))
    y = [np.mean(values[model]) for model in values]
    yerr = [np.std(values[model]) for model in values]
    print yerr


    plt.xticks(x_i, list(values.keys()))
    plt.title(title)
    plt.bar(x_i, y, yerr=yerr)

    plt.show()