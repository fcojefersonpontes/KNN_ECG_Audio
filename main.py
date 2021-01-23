"""coding: utf-8"""

import pandas as pd
from numpy import ones
from numpy import array
from numpy import arange
import math
from scipy import stats
from sklearn.utils import shuffle


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


def normalize_zscore(x):
    if len(x.shape) == 1:
        return stats.zscore(x)

    return [stats.zscore(x[n]) for n in range(x.shape[0])]


def distancia(test, atr_tr):
    return math.sqrt(sum([(test[i] - atr_tr[i]) ** 2 for i in range(len(test))]))


def k_vizinhos(n_k, dists):
    v_dists = dists.copy()
    k_v = [None for n in range(n_k)]
    for i in range(n_k):
        aux = 0
        for j in range(len(v_dists)):
            if v_dists[j] < v_dists[aux]:
                aux = j
        k_v[i] = aux
        v_dists[aux] = max(v_dists)
    return k_v


def classifica(test, atributos_tr, classes_tr, n_k):
    dists = [distancia(test, atr_tr) for atr_tr in atributos_tr]
    k_vi = k_vizinhos(n_k, dists)
    if sum([classes_tr[i] == 1 for i in k_vi]) > sum([classes_tr[i] == 2 for i in k_vi]):
        return 1
    elif sum([classes_tr[i] == 1 for i in k_vi]) < sum([classes_tr[i] == 2 for i in k_vi]):
        return 2
    else:
        return None


if __name__ == '__main__':
    ECG = pd.read_excel("classe1.xlsx", header=None)
    Audio = pd.read_excel("classe2.xlsx", header=None)

    '''Normalização dos sinais (0, 1)'''
    ECG = normalize_min_max(ECG)
    Audio = normalize_min_max(Audio)

    '''columns=['mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis', 'classe']'''
    atributos_ECG = array([mean(ECG), variance(ECG), standard_deviation(ECG), skewness(ECG), kurtosis(ECG)])
    atributos_Audio = array([mean(Audio), variance(Audio), standard_deviation(Audio), skewness(Audio), kurtosis(Audio)])

    '''Normalização dos atributos'''
    '''
    atributos_ECG = array(normalize_zscore(atributos_ECG)).T
    atributos_Audio = array(normalize_zscore(atributos_Audio)).T
    '''
    atributos = ones(100 * 5).reshape(100, 5)
    atributos[:50] = atributos_ECG.T
    atributos[50:] = atributos_Audio.T

    '''Classes dos elementos'''
    classe_ECG = array(ones(ECG.shape[1]))
    classe_Audio = array(ones(Audio.shape[1])) * 2
    classes = array(ones(100))
    classes[:50] = classe_ECG
    classes[50:] = classe_Audio

    '''Embaralha os dados'''
    atributos, classes = shuffle(atributos, classes, random_state=0)

    '''Dados para treino e teste'''
    atributos_teste = array(ones(10 * 5).reshape(10, 5))
    classes_teste = array(ones(10))

    atributos_treino = array(ones(90 * 5).reshape(90, 5))
    classes_treino = array(ones(90))

    '''Valor de K'''
    k = 1

    k_f_results = array(ones(10)) * 0
    '''K-fold com 10 grupos'''
    for k_f in arange(10):

        atributos_teste = atributos[k_f * 10: (k_f + 1) * 10]
        classes_teste = classes[k_f * 10: (k_f + 1) * 10]

        if k_f == 0:
            atributos_treino = atributos[(k_f + 1) * 10:]
            classes_treino = classes[(k_f + 1) * 10:]
        elif k_f == 9:
            atributos_treino = atributos[:k_f * 10]
            classes_treino = classes[:k_f * 10]
        else:
            atributos_treino[:k_f * 10] = atributos[:k_f * 10]
            atributos_treino[k_f * 10:] = atributos[(k_f + 1) * 10:]
            classes_treino[:k_f * 10] = classes[:k_f * 10]
            classes_treino[k_f * 10:] = classes[(k_f + 1) * 10:]

        result = [classifica(tst, atributos_treino, classes_treino, k) for tst in atributos_teste]
        k_f_results[k_f] = sum([False == i for i in classes_teste == result]) / len(classes_teste)

    print('\nK = '+str(k)+'\nK-fold com 10 grupos\n'+'Taxa de erro: '+str(mean(k_f_results)))
