"""coding: utf-8"""

import pandas as pd
from numpy import ones
from numpy import array
import math
from scipy import stats


def length(x):
    if len(x.shape) == 1:
        return len(x)

    return [len(x[n]) for n in range(x.shape[1])]


def mean(x):
    if len(x.shape) == 1:
        return sum(x) / len(x)

    return [sum(x[n]) / x[n].shape[0] for n in range(x.shape[1])]


def variance(x):
    x = x.copy()

    if len(x.shape) == 1:
        mx = mean(x)
        for n in range(x.shape[0]):
            x[n] = (x[n] - mx) ** 2
        return sum(x) / len(x)

    for c in range(x.shape[1]):
        mx = mean(x[c])
        for n in range(x.shape[0]):
            x[c][n] = (x[c][n] - mx) ** 2

    return [sum(x[n]) / x.shape[0] for n in range(x.shape[1])]


def standard_deviation(x):
    if len(x.shape) == 1:
        return math.sqrt(variance(x))

    return [math.sqrt(variance(x[n])) for n in range(x.shape[1])]


def skewness(x):
    if len(x.shape) == 1:
        return sum(((x - mean(x)) / standard_deviation(x)) ** 3) / len(x)

    return [sum(((x[n] - mean(x[n])) / standard_deviation(x[n])) ** 3) / x.shape[0] for n in range(x.shape[1])]


def kurtosis(x):
    if len(x.shape) == 1:
        return sum(((x - mean(x)) / standard_deviation(x)) ** 4) / len(x)

    return [sum(((x[n] - mean(x[n])) / standard_deviation(x[n])) ** 4) / x.shape[0] for n in range(x.shape[1])]


def normalize_min_max(dat):
    x = dat.copy()

    if len(x.shape) == 1:
        return [(x[n] - min(x)) / (max(x) - min(x)) for n in range(len(x))]

    for c in range(x.shape[1]):
        for n in range(x.shape[0]):
            x[c][n] = (dat[c][n] - min(dat[c])) / (max(dat[c]) - min(dat[c]))

    return x


if __name__ == '__main__':
    ECG = pd.read_excel("classe1.xlsx", header=None)
    Audio = pd.read_excel("classe2.xlsx", header=None)

    '''Normalização dos dados (0, 1)'''
    ECG = normalize_min_max(ECG)
    Audio = normalize_min_max(Audio)

    atributos_ECG = pd.DataFrame(array([mean(ECG), variance(ECG), standard_deviation(ECG), skewness(ECG), kurtosis(ECG),
                                        ones(ECG.shape[1])]).T,
                                 columns=['mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis', 'classe'])

    atributos_Audio = pd.DataFrame(array([mean(Audio), variance(Audio), standard_deviation(Audio), skewness(Audio),
                                          kurtosis(Audio), ones(Audio.shape[1]) * 2]).T,
                                   columns=['mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis', 'classe'])

    stats.zscore(atributos_ECG)
    stats.zscore(atributos_Audio)