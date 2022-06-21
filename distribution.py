import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import binom

def underride(d, **options):
    """Aggiunge il valore a d solo se key non è presente in d.
    d: dizionario
    options: argomenti da aggiungere a d
    return: versione modificata di d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d



class Distribution(pd.Series):
    def __init__(self, *args, **kwargs):
        """Inizializza una distribuzione
        Questo codice serve per gestire un comportamento 
        strano della serie Series() e Series([]) hanno dei
        comportamenti differenti.
        Maggiori informazioni : https://github.com/pandas-dev/pandas/issues/16737
        """
        underride(kwargs, name='')
        if args or ('index' in kwargs):
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)

    @property
    def qs(self):
        """ottieni le quantità
        return: array NumPy
        """
        return self.index.values

    @property
    def ps(self):
        """ottengo le probabilità
        return: array Numpy
        """
        return self.values

    def head(self, n=3):
        """Sovrascrivo il metodo Series.head 
        per tornare una Distribution.
        
        n: numero di righe
        returns: Distribution
        """
        s = super().head(n)
        return self.__class__(s)

    def tail(self, n=3):
        """Sovrascrivo il metodo Series.tail 
        per tornare una Distribution.
        
        n: numero di righe
        returns: Distribution
        """
        s = super().tail(n)
        return self.__class__(s)

    def transform(self, *args, **kwargs):
        """Sovrascrivi per lavorare le quantiità, non le probabiltà"""
        qs = self.index.to_series().transform(*args, **kwargs)
        return self.__class__(self.ps, qs, copy=True)

    def _repr_html_(self):

        """Ritorna una rappresentazione HTML delle series
        Usata principalmente per i notebook Jupyter.
        """

        df = pd.DataFrame(dict(probs=self))
        return df._repr_html_()

    def __call__(self, qs):

        """Controllo le quantità.
        """

       
        string_types = (str, bytes, bytearray)

        if hasattr(qs, '__iter__') and not isinstance(qs, string_types):
            s = self.reindex(qs, fill_value=0)
            return s.to_numpy()
        else:
            return self.get(qs, default=0)

    def mean(self):
        """Expected value.
        :return: float
        """
        return self.make_pmf().mean()

    def mode(self, **kwargs):
        """Valore più comune

        Se abbiamo più quantità con la probabilità massima 
        viene tornata la prima quantità massima.

        return: float
        """
        return self.make_pmf().mode(**kwargs)

    def var(self):
        """Ottengo la varianza
        return: float
        """
        return self.make_pmf().var()

    def std(self):
        """Ottengo la deviazione standard
        return: float
        """
        return self.make_pmf().std()

    def median(self):
        """Mediana o il 50° percentile.
        Ci sono molte definizioni di mediana;
        quella implementata qui è solo il 50° percentile

        return: float
        """
        return self.make_cdf().median()

    def quantile(self, ps, **kwargs):
        """Quantili
        Calcola l'inverso del CDF del ps, 
        i valori che corrispondono date le probabilità.

        return: float
        """
        return self.make_cdf().quantile(ps, **kwargs)

    def credible_interval(self, p):
        """Intervallo di credibilità dato un intervallo dato un intervallo
        p: float 0-1
        return: un array di due quantità
        """
        tail = (1 - p) / 2
        ps = [tail, 1 - tail]
        return self.quantile(ps)

    def choice(self, *args, **kwargs):
        """Crea un campione random

        args: le stesse di np.random.choice
        options: le stesse di np.random.choice
        return: array Numpy
        """
        pmf = self.make_pmf()
        return pmf.choice(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Campiona con rimpiazzo usando le probabilità come pesi.
        n: numero di valori
        :return: array Numpy
        """
        cdf = self.make_cdf()
        return cdf.sample(*args, **kwargs)

    def add_dist(self, x):
        """Distribuzione della somma dei valori tratti da self e da x
        x: Distribution, scalare o sequenza
        return: un nuovo oggetto Distribution
        """

        pmf = self.make_pmf()
        res = pmf.add_dist(x)
        return self.make_same(res)

    def sub_dist(self, x):
        pmf = self.make_pmf()
        res = pmf.sub_dist(x)
        return self.make_same(res)

    def mul_dist(self, x):
        pmf = self.make_pmf()
        res = pmf.mul_dist(x)
        return self.make_same(res)

    def div_dist(self, x):
        pmf = self.make_pmf()
        res = pmf.div_dist(x)
        return self.make_same(res)

    def pmf_outer(dist1, dist2, ufunc):
        #TODO: convert other types to Pmf
        pmf1 = dist1
        pmf2 = dist2

        qs = ufunc.outer(pmf1.qs, pmf2.qs)
        ps = np.multiply.outer(pmf1.ps, pmf2.ps)
        return qs * ps

    def gt_dist(self, x):
        pmf = self.make_pmf()
        return pmf.gt_dist(x)

    def lt_dist(self, x):
        pmf = self.make_pmf()
        return pmf.lt_dist(x)

    def ge_dist(self, x):
        pmf = self.make_pmf()
        return pmf.ge_dist(x)

    def le_dist(self, x):
        pmf = self.make_pmf()
        return pmf.le_dist(x)

    def eq_dist(self, x):
        pmf = self.make_pmf()
        return pmf.eq_dist(x)

    def ne_dist(self, x):
        pmf = self.make_pmf()
        return pmf.ne_dist(x)

    def max_dist(self, n):
        cdf = self.make_cdf().max_dist(n)
        return self.make_same(cdf)

    def min_dist(self, n):
        cdf = self.make_cdf().min_dist(n)
        return self.make_same(cdf)

class Pmf(Distribution):
    """Represents a probability Mass Function (PMF)."""

    def copy(self, deep=True):
        """Make a copy.
        :return: new Pmf
        """
        return Pmf(self, copy=deep)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Pmf.
        :return: Pmf
        """
        return self

    # Pmf overrides the arithmetic operations in order
    # to provide fill_value=0 and return a Pmf.

    def add(self, x, **kwargs):
        """Override add to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.add(self, x, **kwargs)
        return Pmf(s)

    __add__ = add
    __radd__ = add

    def sub(self, x, **kwargs):
        """Override the - operator to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.subtract(self, x, **kwargs)
        return Pmf(s)

    __sub__ = sub
    __rsub__ = sub

    def mul(self, x, **kwargs):
        """Override the * operator to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.multiply(self, x, **kwargs)
        return Pmf(s)

    __mul__ = mul
    __rmul__ = mul

    def div(self, x, **kwargs):
        """Override the / operator to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.divide(self, x, **kwargs)
        return Pmf(s)

    __div__ = div
    __rdiv__ = div
    __truediv__ = div
    __rtruediv__ = div

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).
        :return: normalizing constant
        """
        total = self.sum()
        self /= total
        return total

    def mean(self):
        """Computes expected value.
        :return: float
        """
        # TODO: error if not normalized
        # TODO: error if the quantities are not numeric
        return np.sum(self.ps * self.qs)

    def mode(self, **kwargs):
        """Most common value.
        If multiple quantities have the maximum probability,
        the first maximal quantity is returned.
        :return: float
        """
        return self.idxmax(**kwargs)

    def var(self):
        """Variance of a PMF.
        :return: float
        """
        m = self.mean()
        d = self.qs - m
        return np.sum(d ** 2 * self.ps)

    def std(self):
        """Standard deviation of a PMF.
        :return: float
        """
        return np.sqrt(self.var())

    def choice(self, *args, **kwargs):
        """Makes a random sample.
        Uses the probabilities as weights unless `p` is provided.
        args: same as np.random.choice
        kwargs: same as np.random.choice
        :return: NumPy array
        """
        underride(kwargs, p=self.ps)
        return np.random.choice(self.qs, *args, **kwargs)

    def add_dist(self, x):
        """Computes the Pmf of the sum of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.add.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs + x)

    def sub_dist(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.subtract.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs - x)

    def mul_dist(self, x):
        """Computes the Pmf of the product of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.multiply.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs * x)

    def div_dist(self, x):
        """Computes the Pmf of the ratio of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.divide.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs / x)

    def convolve_dist(self, dist, ufunc):
        """Convolve two distributions.
        dist: Distribution
        ufunc: elementwise function for arrays
        :return: new Pmf
        """
        dist = dist.make_pmf()
        qs = ufunc(self.qs, dist.qs).flatten()
        ps = np.multiply.outer(self.ps, dist.ps).flatten()
        series = pd.Series(ps).groupby(qs).sum()

        return Pmf(series)

    def gt_dist(self, x):
        """Probability that a value from pmf1 is greater than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.greater).sum()
        else:
            return self[self.qs > x].sum()

    def lt_dist(self, x):
        """Probability that a value from pmf1 is less than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.less).sum()
        else:
            return self[self.qs < x].sum()

    def ge_dist(self, x):
        """Probability that a value from pmf1 is >= than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.greater_equal).sum()
        else:
            return self[self.qs >= x].sum()

    def le_dist(self, x):
        """Probability that a value from pmf1 is <= than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.less_equal).sum()
        else:
            return self[self.qs <= x].sum()

    def eq_dist(self, x):
        """Probability that a value from pmf1 equals a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.equal).sum()
        else:
            return self[self.qs == x].sum()

    def ne_dist(self, x):
        """Probability that a value from pmf1 is <= than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.not_equal).sum()
        else:
            return self[self.qs != x].sum()

    def pmf_outer(self, dist, ufunc):
        """Computes the outer product of two PMFs.
        dist: Distribution object
        ufunc: function to apply to the qs
        :return: NumPy array
        """
        dist = dist.make_pmf()
        qs = ufunc.outer(self.qs, dist.qs)
        ps = np.multiply.outer(self.ps, dist.ps)
        return qs * ps

    def bar(self, **options):
        """Make a bar plot.
        options: passed to pd.Series.plot
        """
        underride(options, rot=0)
        self.plot.bar(**options)

    def make_joint(self, other, **options):
        """Make joint distribution (assuming independence).
        :param self:
        :param other:
        :param options: passed to Pmf constructor
        :return: new Pmf
        """
        qs = pd.MultiIndex.from_product([self.qs, other.qs])
        ps = np.multiply.outer(self.ps, other.ps).flatten()
        return Pmf(ps, index=qs, **options)

    def marginal(self, i, name=None):
        """Gets the marginal distribution of the indicated variable.
        i: index of the variable we want
        name: string
        :return: Pmf
        """
        return Pmf(self.sum(level=i))

    def conditional(self, i, val, name=None):
        """Gets the conditional distribution of the indicated variable.
        i: index of the variable we're conditioning on
        val: the value the ith variable has to have
        name: string
        :return: Pmf
        """
        pmf = Pmf(self.xs(key=val, level=i), copy=True)
        pmf.normalize()
        return pmf

    def update(self, likelihood, data):
        """Bayesian update.
        likelihood: function that takes (data, hypo) and returns
                    likelihood of data under hypo, P(data|hypo)
        data: in whatever format likelihood understands
        :return: normalizing constant
        """
        for hypo in self.qs:
            self[hypo] *= likelihood(data, hypo)

        return self.normalize()

    def max_prob(self):
        """Value with the highest probability.
        :return: the value with the highest probability
        """
        return self.idxmax()

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Pmf.
        :return: Cdf
        """
        normalize = kwargs.pop('normalize', False)

        cumulative = np.cumsum(self)
        cdf = Cdf(cumulative, self.index.copy(), **kwargs)

        if normalize:
            cdf.normalize()

        return cdf

    def make_surv(self, **kwargs):
        """Make a Surv from the Pmf.
        :return: Surv
        """
        cdf = self.make_cdf()
        return cdf.make_surv(**kwargs)


    def make_same(self, dist):
        """Convert the given dist to Pmf
        :param dist:
        :return: Pmf
        """
        return dist.make_pmf()

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, ascending=True,
                 dropna=True, na_position='last', **options):
        """Make a PMF from a sequence of values.
        seq: iterable
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        ascending: whether to sort in ascending order, default True
        dropna: whether to drop NaN values, default True
        na_position: If ‘first’ puts NaNs at the beginning,
                        ‘last’ puts NaNs at the end.
        options: passed to the pd.Series constructor
        NOTE: In the current implementation, `from_seq` sorts numerical
           quantities whether you want to or not.  If keeping
           the order of the elements is important, let me know and
           I'll rethink the implementation
        :return: Pmf object
        """
        # compute the value counts
        series = pd.Series(seq).value_counts(normalize=normalize,
                                             sort=False,
                                             dropna=dropna)
        # make the result a Pmf
        # (since we just made a fresh Series, there is no reason to copy it)
        options['copy'] = False
        pmf = Pmf(series, **options)

        # sort in place, if desired
        if sort:
            pmf.sort_index(inplace=True,
                           ascending=ascending,
                           na_position=na_position)

        return pmf


class Cdf(Distribution):
    """Represents a Cumulative Distribution Function (CDF)."""

    def copy(self, deep=True):
        """Make a copy.
        :return: new Cdf
        """
        return Cdf(self, copy=deep)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a CDF from a sequence of values.
        seq: iterable
        normalize: whether to normalize the Cdf, default True
        sort: whether to sort the Cdf by values, default True
        options: passed to the pd.Series constructor
        :return: CDF object
        """
        # if normalize==True, normalize AFTER making the Cdf
        # so the last element is exactly 1.0
        pmf = Pmf.from_seq(seq, normalize=False, sort=sort, **options)
        return pmf.make_cdf(normalize=normalize)

    def step(self, **options):
        """Plot the Cdf as a step function.
        :param options: passed to pd.Series.plot
        :return:
        """
        underride(options, drawstyle="steps-post")
        self.plot(**options)

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).
        :return: normalizing constant
        """
        total = self.ps[-1]
        self /= total
        return total

    @property
    def forward(self, **kwargs):
        """Compute the forward Cdf
        :param kwargs: keyword arguments passed to interp1d
        :return interpolation function from qs to ps
        """

        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, 1),
        )

        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse Cdf
        :param kwargs: keyword arguments passed to interp1d
        :return: interpolation function from ps to qs
        """
        underride(
            kwargs,
            kind="next",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(self.qs[0], np.nan),
        )

        interp = interp1d(self.ps, self.qs, **kwargs)
        return interp

    # calling a Cdf like a function does forward lookup
    __call__ = forward

    # quantile is the same as an inverse lookup
    quantile = inverse

    def median(self):
        """Median (50th percentile).
        :return: float
        """
        return self.quantile(0.5)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Cdf.
        :param normalize: Boolean, whether to normalize the Pmf
        :return: Pmf
        """
        #TODO: check for consistent behavior of copy flag for all make_x
        normalize = kwargs.pop('normalize', False)

        diff = np.diff(self, prepend=0)
        pmf = Pmf(diff, index=self.index.copy(), **kwargs)
        if normalize:
            pmf.normalize()
        return pmf


    def make_same(self, dist):
        """Convert the given dist to Cdf
        :param dist:
        :return: Cdf
        """
        return dist.make_cdf()

    def sample(self, n=1):
        """Samples with replacement using probabilities as weights.
        n: number of values
        :return: NumPy array
        """
        ps = np.random.random(n)
        return self.inverse(ps)

    def max_dist(self, n):
        """Distribution of the maximum of `n` values from this distribution.
        n: integer
        :return: Cdf
        """
        ps = self**n
        return Cdf(ps, self.index.copy())

    def min_dist(self, n):
        """Distribution of the minimum of `n` values from this distribution.
        n: integer
        :return: Cdf
        """
        ps = 1 - (1 - self)**n
        return Cdf(ps, self.index.copy())
        
def make_binomial(n, p):
    """Make a binomial distribution.
    n: number of trials
    p: probability of success
    returns: Pmf representing the distribution of k
    """
    ks = np.arange(n+1)
    ps = binom.pmf(ks, n, p)
    return Pmf(ps, ks)
    
def make_mixture(pmf, pmf_seq):
    """Make a mixture of distributions.
    pmf: mapping from each hypothesis to its probability
         (or it can be a sequence of probabilities)
    pmf_seq: sequence of Pmfs, each representing
             a conditional distribution for one hypothesis
    returns: Pmf representing the mixture
    """
    df = pd.DataFrame(pmf_seq).fillna(0).transpose()
    df *= np.array(pmf)
    total = df.sum(axis=1)
    return Pmf(total)