import re
import pandas as pd
import numpy as np
from collections import defaultdict

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
