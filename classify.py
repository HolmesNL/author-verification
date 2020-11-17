#!/usr/bin/env python3

import argparse
import collections
import logging
import os
import warnings

import confidence
import lir
import lir.transformers
from matplotlib import pyplot as plt
import numpy as np
import scipy.spatial
import scipy.stats
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from tqdm import tqdm

from authorship import get_data_replacement as get_data
import experiments
import Function_file as data


DEFAULT_LOGLEVEL = logging.WARNING
LOG = logging.getLogger(__name__)


def setupLogging(args):
    loglevel = max(logging.DEBUG, min(logging.CRITICAL, DEFAULT_LOGLEVEL + (args.q - args.v) * 10))

    # setup formatter
    log_format = '[%(asctime)-15s %(levelname)s] %(name)s: %(message)s'
    fmt = logging.Formatter(log_format)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(loglevel)
    logging.getLogger().addHandler(ch)

    # setup a file handler
    fh = logging.FileHandler('run.log', mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    logging.getLogger('').setLevel(logging.DEBUG)


class ParticleCountToFraction(sklearn.base.TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return (X.T / np.sum(X, axis=1)).T


class KdeCdfTransformer(sklearn.base.TransformerMixin):
    def __init__(self, value_range=(None, None), resolution=1000, plot_cdf=False):
        self._value_range = value_range
        self._resolution = resolution
        self._kernels = None
        self._plot_cdf = plot_cdf

    def get_range(self, feature_values):
        lower = self._value_range[0] or np.min(feature_values)
        upper = self._value_range[1] or np.max(feature_values)

        return lower, upper

    def fit(self, X):
        assert len(X.shape) == 2

        self._kernels = []
        for i in range(X.shape[1]):
            feature_values = X[:,i]
            lower, upper = self.get_range(feature_values)

            kernel = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=.1).fit(feature_values.reshape(-1, 1))
            precomputed_values = np.arange(self._resolution+1).reshape(-1, 1) / self._resolution * (upper-lower) + lower
            density = np.exp(kernel.score_samples(precomputed_values))
            cumulative_density = np.cumsum(density)
            cumulative_density = cumulative_density / cumulative_density[-1]
            self._kernels.append(cumulative_density)

            if self._plot_cdf:
                plt.plot(precomputed_values, cumulative_density)

        if self._plot_cdf:
            plt.show()

        return self

    def transform(self, X):
        assert self._kernels is not None
        assert len(X.shape) == 2
        assert X.shape[1] == len(self._kernels)

        features = []
        for i in range(X.shape[1]):
            feature_values = X[:,i]
            lower, upper = self.get_range(feature_values)

            percentiles = self._kernels[i][((feature_values - lower) / (upper-lower) * self._resolution).astype(int)]
            features.append(percentiles)

        return np.stack(features, axis=1)


class GaussianCdfTransformer(sklearn.base.TransformerMixin):
    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

        self._valid_features = self._std > 0
        self._mean = self._mean[self._valid_features]
        self._std = self._std[self._valid_features]

        return self

    def transform(self, X):
        assert len(X.shape) == 2
        X = X[:,self._valid_features]
        return scipy.stats.norm.cdf(X, self._mean, self._std)


GaussParams = collections.namedtuple('StandardParams', ['mean0', 'std0', 'mean1', 'std1'])
class GaussianScorer(sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        assert np.all(np.arange(np.max(y)+1) == np.unique(y)), 'classes must be numbered 0..n and all classes must occur at least once'

        self._classes = np.unique(y)
        self._models = {}
        for cls in self._classes:
            X0 = X[y!=cls]
            X1 = X[y==cls]

            mean0 = np.mean(X0, axis=0)
            std0 = np.std(X0, axis=0)
            mean1 = np.mean(X1, axis=0)
            std1 = np.std(X1, axis=0)

            # if parameters could not be estimated, assume std 1
            std0[std0 == 0] = 1
            std1[std1 == 0] = 1

            self._models[cls] = GaussParams(mean0, std0, mean1, std1)

    def predict_proba(self, X):
        return lir.util.to_probability(self.predict_lr(X))

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def predict_lr(self, X):
        p = []
        for cls in self._classes:
            params = self._models[cls]
            p0 = scipy.stats.norm.pdf(X, params.mean0, params.std0)
            p1 = scipy.stats.norm.pdf(X, params.mean1, params.std1)
            with np.errstate(divide='ignore'):
                p.append(p1/(p0+p1))

        return np.prod(lir.to_odds(np.array(p)), axis=2).T  # multiply the odds over categories (assume conditional independence)


class AbsDiffTransformer(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        return np.abs(X[:,:,0] - X[:,:,1])


class BrayDistance(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        left = X[:,:,0]
        right = X[:,:,1]

        return np.abs(right - left) / (np.abs(right + left) + 1)


class ShanDistanceVector(sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        p = X[:,:,0]
        q = X[:,:,1]
        m = (p + q) / 2.0

        # TODO: may be negative
        left = scipy.spatial.distance.rel_entr(p, m)
        right = scipy.spatial.distance.rel_entr(q, m)
        result = np.sqrt((left + right) / 2.0)

        assert X.shape[0:2] == result.shape
        return result


class VectorDistance(sklearn.base.TransformerMixin):
    def __init__(self, dfunc):
        self._dfunc = dfunc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        distance_by_pair = []
        for z in range(X.shape[0]):
            dist = self._dfunc(X[z,:,0], X[z,:,1])
            distance_by_pair.append(dist)

        return np.array(distance_by_pair).reshape(-1, 1)

    def predict_proba(self, X):
        p0 = self.transform(X)
        return np.stack([p0, 1/p0], axis=1)


class makeplots:
    def __init__(self, path_prefix=None):
        self.path_prefix = path_prefix

    def __call__(self, lrs, y, title='', shortname=''):
        n_same = int(np.sum(y))
        n_diff = int(y.size-np.sum(y))

        LOG.info(f'  counts by class: diff={n_diff}; same={n_same}')
        LOG.info(f'  average LR by class: 1/{np.exp(np.mean(-np.log(lrs[y==0])))}; {np.exp(np.mean(np.log(lrs[y==1])))}')
        LOG.info(f'  cllr: {lir.metrics.cllr(lrs, y)}')

        path_prefix = os.path.join(self.path_prefix, shortname)
        tippet_path = f'{path_prefix}_tippet.png' if self.path_prefix is not None else None
        pav_path = f'{path_prefix}_pav.png' if self.path_prefix is not None else None
        ece_path = f'{path_prefix}_ece.png' if self.path_prefix is not None else None

        kw_figure = {}

        lir.plotting.plot_tippett(lrs, y, savefig=tippet_path, kw_figure=kw_figure)
        lir.plotting.plot_pav(lrs, y, savefig=pav_path, kw_figure=kw_figure)
        lir.ece.plot(lrs, y, path=ece_path, on_screen=not ece_path, kw_figure=kw_figure)


def get_pairs(X, y, authors_subset, sample_size):
    X_subset = X[np.isin(y, authors_subset), :]
    y_subset = y[np.isin(y, authors_subset)]

    # pair instances: same source and different source
    return lir.transformers.InstancePairing(same_source_limit=sample_size//2, different_source_limit=sample_size//2).transform(X_subset, y_subset)


def get_batch_simple(X, y, repeats):
    for i in range(repeats):
        authors = np.unique(y)
        authors_train, authors_test = sklearn.model_selection.train_test_split(authors, test_size=.1, random_state=i)

        X_train, y_train = get_pairs(X, y, authors_train, 1000)
        X_test, y_test = get_pairs(X, y, authors_test, 1000)

        yield X_train, y_train, X_test, y_test


def evaluate_samesource(desc, frida_path, n_frequent_words, tokens_per_sample, preprocessor, classifier, calibrator, plot=None, repeats=1):
    """
    Run an experiment with the parameters provided.

    :param desc: free text description of the experiment
    :param frida_path: path to transcript index file
    :param n_frequent_words: int: number of most frequent words
    :param tokens_per_sample: int: number of tokens per sample (sample length)
    :param preprocessor: Pipeline: an sklearn pipeline
    :param classifier: Pipeline: an sklearn pipeline with a classifier as last element
    :param calibrator: a LIR calibrator
    :param plot: a plotting function
    :param repeats: int: the number of times the experiment is run
    :return: a CLLR
    """
    #calibrator = lir.plotting.PlottingCalibrator(calibrator, lir.plotting.plot_score_distribution_and_calibrator_fit)
    clf = lir.CalibratedScorer(classifier, calibrator)

    desc_pre = '; '.join(name for name, tr in preprocessor.steps)
    desc_clf = '; '.join(name for name, tr in clf.scorer.steps)
    title = f'{desc}: using common source model: {desc_pre}; {desc_clf}; {ds}; repeats={repeats}'
    LOG.info(title)
    LOG.info(f'{desc}: number of classes: {np.unique(ds.get()[1]).size}')
    LOG.info(f'{desc}: number of instances: {ds.get()[1].size}')

    X, y = get_data(frida_path, n_frequent_words=n_frequent_words, tokens_per_sample=tokens_per_sample)
    assert X.shape[0] > 0

    X = preprocessor.fit_transform(X)

    lrs = []
    y_all = []
    for X_train, y_train, X_test, y_test in tqdm(get_batch_simple(X, y, repeats)):
        # fit a classifier and calculate LRs
        clf.fit(X_train, y_train)
        lrs.append(clf.predict_lr(X_test))
        y_all.append(y_test)

    lrs = np.concatenate(lrs)
    y_all = np.concatenate(y_all)

    if plot is not None:
        plot(lrs, y_all, title=title, shortname=desc)

    return lir.metrics.cllr(lrs, y_all)


def aggregate_results(results):
    for params, result in results:
        desc = ', '.join(f'{name}={value}' for name, value in params)
        print(f'{desc}: cllr={result}')
    

def run(frida_path, resultdir):
    ### PREPROCESSORS

    prep_none = sklearn.pipeline.Pipeline([
            ('scale:none', None),
            ('pop:none', None),
        ])

    prep_std = sklearn.pipeline.Pipeline([
            ('scale:standard', sklearn.preprocessing.StandardScaler()),
            ('pop:none', None),
        ])

    prep_norm = sklearn.pipeline.Pipeline([
            ('scale:norm', sklearn.preprocessing.Normalizer()),
            ('pop:none', None),
        ])

    prep_sum = sklearn.pipeline.Pipeline([
            ('scale:sum', ParticleCountToFraction()),
            ('pop:none', None),
        ])

    prep_gauss = sklearn.pipeline.Pipeline([
            ('scale:standard', sklearn.preprocessing.StandardScaler()),
            ('pop:gauss', GaussianCdfTransformer()),  # cumulative density function for each feature
            #('pop:gauss', sklearn.preprocessing.QuantileTransformer()),  # cumulative density function for each feature
        ])

    prep_kde = sklearn.pipeline.Pipeline([
            ('scale:standard', sklearn.preprocessing.StandardScaler()),
            ('pop:kde', KdeCdfTransformer()),  # cumulative density function for each feature
        ])

    ### CLASSIFIERS

    dist = sklearn.pipeline.Pipeline([
            ('dist:shan', VectorDistance(scipy.spatial.distance.jensenshannon)),
        ])

    logit = sklearn.pipeline.Pipeline([
            ('diff:abs', AbsDiffTransformer()),
            #('shan', ShanDistance()),
            #('bray', BrayDistance()),
            ('clf:logit', LogisticRegression(class_weight='balanced')),
        ])

    svc = sklearn.pipeline.Pipeline([
            ('diff:shan', ShanDistanceVector()),
            ('clf:svc', sklearn.svm.SVC(gamma='scale', kernel='linear', probability=True, class_weight='balanced')),
        ])

    exp = experiments.Evaluation(evaluate_samesource, aggregate_results)

    exp.parameter('frida_path', frida_path)
    exp.parameter('plot', makeplots(resultdir))

    exp.parameter('n_frequent_words', 200)
    exp.addSearch('n_frequent_words', [50, 150, 250], include_default=False)

    exp.parameter('tokens_per_sample', 250)
    exp.addSearch('tokens_per_sample', [250, 750, 1500], include_default=False)

    exp.parameter('preprocessor', prep_gauss)

    exp.parameter('classifier', ('logit', logit))
    exp.addSearch('classifier', [('dist', dist), ('svc', svc)])

    exp.parameter('calibrator', lir.KDECalibrator(bandwidth=.3))
    exp.parameter('repeats', 5)

    try:
        #exp.runDefaults()
        exp.runSearch('tokens_per_sample')
        #exp.runFullGrid(['n_frequent_words', 'tokens_per_sample', 'classifier'])
    except Exception as e:
        LOG.fatal(str(e))
        raise


if __name__ == '__main__':
    config = confidence.load_name('authorship', 'local')
    warnings.filterwarnings("error")
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    parser.add_argument('--frida-path', help='path to FRIDA transcripts', default=config.frida_path)
    parser.add_argument('--resultdir', help='path to generated output files', default=config.resultdir)
    args = parser.parse_args()

    setupLogging(args)

    os.makedirs(args.resultdir, exist_ok=True)
    run(args.frida_path, args.resultdir)
