import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt

from distribution import Pmf

Gray20 = (0.162, 0.162, 0.162, 0.7)
Gray30 = (0.262, 0.262, 0.262, 0.7)
Gray40 = (0.355, 0.355, 0.355, 0.7)
Gray50 = (0.44, 0.44, 0.44, 0.7)
Gray60 = (0.539, 0.539, 0.539, 0.7)
Gray70 = (0.643, 0.643, 0.643, 0.7)
Gray80 = (0.757, 0.757, 0.757, 0.7)
Pu20 = (0.247, 0.0, 0.49, 0.7)
Pu30 = (0.327, 0.149, 0.559, 0.7)
Pu40 = (0.395, 0.278, 0.62, 0.7)
Pu50 = (0.46, 0.406, 0.685, 0.7)
Pu60 = (0.529, 0.517, 0.742, 0.7)
Pu70 = (0.636, 0.623, 0.795, 0.7)
Pu80 = (0.743, 0.747, 0.866, 0.7)
Bl20 = (0.031, 0.188, 0.42, 0.7)
Bl30 = (0.031, 0.265, 0.534, 0.7)
Bl40 = (0.069, 0.365, 0.649, 0.7)
Bl50 = (0.159, 0.473, 0.725, 0.7)
Bl60 = (0.271, 0.581, 0.781, 0.7)
Bl70 = (0.417, 0.681, 0.838, 0.7)
Bl80 = (0.617, 0.791, 0.882, 0.7)
Gr20 = (0.0, 0.267, 0.106, 0.7)
Gr30 = (0.0, 0.312, 0.125, 0.7)
Gr40 = (0.001, 0.428, 0.173, 0.7)
Gr50 = (0.112, 0.524, 0.253, 0.7)
Gr60 = (0.219, 0.633, 0.336, 0.7)
Gr70 = (0.376, 0.73, 0.424, 0.7)
Gr80 = (0.574, 0.824, 0.561, 0.7)
Or20 = (0.498, 0.153, 0.016, 0.7)
Or30 = (0.498, 0.153, 0.016, 0.7)
Or40 = (0.599, 0.192, 0.013, 0.7)
Or50 = (0.746, 0.245, 0.008, 0.7)
Or60 = (0.887, 0.332, 0.031, 0.7)
Or70 = (0.966, 0.475, 0.147, 0.7)
Or80 = (0.992, 0.661, 0.389, 0.7)
Re20 = (0.404, 0.0, 0.051, 0.7)
Re30 = (0.495, 0.022, 0.063, 0.7)
Re40 = (0.662, 0.062, 0.085, 0.7)
Re50 = (0.806, 0.104, 0.118, 0.7)
Re60 = (0.939, 0.239, 0.178, 0.7)
Re70 = (0.985, 0.448, 0.322, 0.7)
Re80 = (0.988, 0.646, 0.532, 0.7)

color_list = [Bl30, Or70, Gr50, Re60, Pu20, Gray70, Re80, Gray50, 
              Gr70, Bl50, Re40, Pu70, Or50, Gr30, Bl70, Pu50, Gray30]
color_cycle = cycler(color=color_list)

def set_pyplot_params():
    plt.rcParams['axes.prop_cycle'] = color_cycle
    plt.rcParams['lines.linewidth'] = 3


def decorate(**options):
    """Decorate the current axes.
    
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
             
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()

def readStataDct(dct_file, **options):
    type_map = dict(byte=int, int=int, long=int, float=float, 
                    double=float, numeric=float)

    var_info = []
    with open(dct_file, **options) as f:
        for line in f:
            match = re.search( r'_column\(([^)]*)\)', line)
            if not match:
                continue
            start = int(match.group(1)) - 1
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            length = int(re.findall("\d+", fstring)[0])
            var_info.append((start,length, vtype, name, fstring, long_desc))
        
        columns = ['start','length', 'type', 'name', 'fstring', 'desc']
        variables = pd.DataFrame(var_info, columns=columns)
        variables['end'] = variables['start'] + variables['length']
    return variables


def readStataDataset(filename,variables, **options):
    colspecs = variables[['start', 'end']]
    colspecs = colspecs.astype(np.int32).values.tolist()
    names = variables['name']
    
    df = pd.read_fwf(filename,colspecs=colspecs, names=names,**options)
    
    return df

def cleanFemPreg(df):
    df.agepreg /= 100
    na_vals = [97, 98, 99]
    df.birthwgt_lb.replace(na_vals,np.nan,inplace=True)
    df.birthwgt_oz.replace(na_vals,np.nan,inplace=True)

    df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan
        
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0

def readReadFemPreg():
    cols = readStataDct('./data/2002FemPreg.dct',encoding='iso-8859-1')
    df = readStataDataset('./data/2002FemPreg.dat.gz',cols,compression='gzip')

    cleanFemPreg(df)

    return df

def readFemResp():
    cols = readStataDct('./data/2002FemResp.dct',encoding='iso-8859-1')
    df = readStataDataset('./data/2002FemResp.dat.gz',cols,compression='gzip')

    return df

def readBabyBoom(filename='./data/babyboom.dat'):
    var_info = [
        ('time', 1, 8, int),
        ('sex', 9, 16, int),
        ('weight_g', 17, 24, int),
        ('minutes', 25, 32, int),
        ]
    columns = ['name', 'start', 'end', 'type']
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1

    colspecs = variables[['start', 'end']]
    colspecs = colspecs.astype(np.int32).values.tolist()
    names = variables['name']

    df = pd.read_fwf(filename,colspecs=colspecs, names=names,skiprows=59)
    return df

def cleanBrfssFrame(df):
    """Recodes BRFSS variables.

    df: DataFrame
    """
    # clean age
    df.age.replace([7, 9], float('NaN'), inplace=True)

    # clean height
    df.htm3.replace([999], float('NaN'), inplace=True)

    # clean weight
    df.wtkg2.replace([99999], float('NaN'), inplace=True)
    df.wtkg2 /= 100.0

    # clean weight a year ago
    df.wtyrago.replace([7777, 9999], float('NaN'), inplace=True)
    df['wtyrago'] = df.wtyrago.apply(lambda x: x/2.2 if x < 9000 else x-9000)

def readBrfss(filename='./data/CDBRFS08.ASC.gz'):
    """Reads the BRFSS data.

    filename: string
    compression: string
    nrows: int number of rows to read, or None for all

    returns: DataFrame
    """
    var_info = [
        ('age', 101, 102, int),
        ('sex', 143, 143, int),
        ('wtyrago', 127, 130, int),
        ('finalwt', 799, 808, int),
        ('wtkg2', 1254, 1258, int),
        ('htm3', 1251, 1253, int),
        ]
    columns = ['name', 'start', 'end', 'type']
    variables = pd.DataFrame(var_info, columns=columns)
    variables.end += 1
    
    colspecs = variables[['start', 'end']] - 1
    colspecs = colspecs.astype(np.int32).values.tolist()
    names = variables['name']

    df = pd.read_fwf(filename,colspecs=colspecs, names=names,compression='gzip')

    cleanBrfssFrame(df)
    return df
    
    
def readPopulation(filename='./data/PEP_2012_PEPANNRES_with_ann.csv'):
    df = pd.read_csv(filename, header=None, skiprows=2,
                         encoding='iso-8859-1')
    populations = df[7]
    populations.replace(0, np.nan, inplace=True)
    return populations.dropna()
    

def calcPmf(data):
    result = defaultdict(int)

    data_sorted = np.sort(data)
    unique, counts = np.unique(data_sorted, return_counts=True)
    pmf = counts / len(data_sorted)
    
    for val,freq in zip(unique,pmf):
        result[val] = freq
    
    return result


def calcCdf(data):
    data_sorted = np.sort(data)
    unique, counts = np.unique(data_sorted, return_counts=True)
    pmf = counts / len(data_sorted)
    cdf = np.cumsum(pmf)
    return unique, pmf, cdf
    
def percentile(unique,cdf,ps=None):
    
    if ps is None:
        return unique
    
    ps = np.asarray(ps)
    if np.any(ps < 0) or np.any(ps > 1):
        raise ValueError('Probability p must be in range [0, 1]')
    
    index = np.searchsorted(cdf, ps, side='left')
    return unique[index]
    
def confidenceInterval(unique,cdf,ps=None):
    
    if ps is None:
        return unique
    
    ps = np.asarray(ps)
    if np.any(ps < 0) or np.any(ps > 1):
        raise ValueError('Probability p must be in range [0, 1]')
    
    prob = (1 - ps) / 2
    
    intervalMin = np.searchsorted(cdf, prob, side='left')
    intervalMax = np.searchsorted(cdf, 1 - prob, side='left')
    return unique[intervalMin],unique[intervalMax]
    
def var(xs, mu=None, ddof=0):
    xs = np.asarray(xs)

    if mu is None:
        mu = xs.mean()

    ds = xs - mu
    return np.dot(ds, ds) / (len(xs) - ddof)    
    
def meanVar(xs, ddof=0):
    xs = np.asarray(xs)
    mean = xs.mean()
    s2 = var(xs, mean, ddof)
    return mean, s2    
    
    
def cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


def corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = meanVar(xs)
    meany, vary = meanVar(ys)

    corr = cov(xs, ys, meanx, meany) / math.sqrt(varx * vary)

    return corr
    
    
def make_die(sides):
    outcomes = np.arange(1, sides+1)
    die = Pmf(1/sides, outcomes)
    return die
    
def add_dist_seq(seq):
    """Compute Pmf of the sum of values from seq."""
    total = seq[0]
    for other in seq[1:]:
        total = total.add_dist(other)
    return total

def make_joint(s1, s2):
    """Compute the outer product of two Series.
    First Series goes across the columns;
    second goes down the rows.
    s1: Series
    s2: Series
    return: DataFrame
    """
    X, Y = np.meshgrid(s1, s2)
    return pd.DataFrame(X*Y, columns=s1.index, index=s2.index)

def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.
    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def plot_contour(joint, **options):
    """Plot a joint distribution.
    joint: DataFrame representing a joint PMF
    """
    low = joint.to_numpy().min()
    high = joint.to_numpy().max()
    levels = np.linspace(low, high, 6)
    levels = levels[1:]

    underride(options, levels=levels, linewidths=1)
    cs = plt.contour(joint.columns, joint.index, joint, **options)
    decorate(xlabel=joint.columns.name,
             ylabel=joint.index.name)
    return cs

def normalize(joint):
    """Normalize a joint distribution.
    joint: DataFrame
    """
    prob_data = joint.to_numpy().sum()
    joint /= prob_data
    return prob_data
