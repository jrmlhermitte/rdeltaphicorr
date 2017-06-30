import numpy as np
from rdpc import _running_var_avg

from numpy.testing import assert_almost_equal, assert_array_almost_equal


def test_runavg_stats():
    # verify that the running average is working properly
    # this is crucial since the delta phi correlations use this
    # create 100 elements
    mean = 10
    sigma = 2
    ndata = 10000
    np.random.seed(1431423)
    data = np.random.normal(loc=10., scale=2, size=(ndata))

    res = _running_var_avg(data)
    # estimated mean and M2
    n, emean, eM2 = res
    esigma = np.sqrt(eM2/(n-1))

    # 1/np.sqrt(ndata) for ndata =10^4 is just at 2 decimal precision
    # so let's just choose 1 decimal
    assert_almost_equal(emean, mean, decimal=1)
    assert_almost_equal(esigma, sigma, decimal=1)


def test_runavg_0dim():
    ''' test 0 dim data. '''
    np.random.seed(1433)
    # singleton of data
    data = [1, 1, 1, 1]

    # previous state
    res = None
    for datum in data:
        res = _running_var_avg(datum, prev=res)
    # estimated mean and M2

    n, emean, eM2 = res
    esigma = np.sqrt(eM2/(n-1))

    # 1/np.sqrt(ndata) for ndata =10^4 is just at 2 decimal precision
    # so let's just choose 1 decimal
    assert_array_almost_equal(emean, np.array([1]), decimal=1)
    assert_array_almost_equal(esigma, np.array([0]), decimal=1)


def test_runavg_1dim():
    ''' test 1 dim data. '''
    np.random.seed(1433)
    # singleton of data
    data = np.ones((10, 10))

    # previous state
    res = None
    # if iterating, need to fake an extra dimension
    for subdata in data:
        res = _running_var_avg(subdata[np.newaxis, :], prev=res)
    # estimated mean and M2

    n, emean, eM2 = res
    esigma = np.sqrt(eM2/(n-1))

    # 1/np.sqrt(ndata) for ndata =10^4 is just at 2 decimal precision
    # so let's just choose 1 decimal
    assert_array_almost_equal(emean, np.ones(10), decimal=1)
    assert_almost_equal(esigma, np.zeros((10)), decimal=1)

    # alternate way:
    res = _running_var_avg(data, prev=res)
    n, emean, eM2 = res
    esigma = np.sqrt(eM2/(n-1))
    assert_array_almost_equal(emean, np.ones(10), decimal=1)
    assert_almost_equal(esigma, np.zeros((10)), decimal=1)


def test_runavg_2dim():
    ''' test 2 dim data.
        Doesn't test the stats
    '''
    np.random.seed(1433)
    # singleton of data
    data = np.ones((10, 10, 10))

    # previous state
    res = None
    for subdata in data:
        res = _running_var_avg(subdata[np.newaxis, :, :], prev=res)
    # estimated mean and M2

    n, emean, eM2 = res
    esigma = np.sqrt(eM2/(n-1))

    # 1/np.sqrt(ndata) for ndata =10^4 is just at 2 decimal precision
    # so let's just choose 1 decimal
    assert_array_almost_equal(emean, np.ones((10, 10)), decimal=1)
    assert_almost_equal(esigma, np.zeros((10, 10)), decimal=1)

    # alternate way:
    res = _running_var_avg(data, prev=res)
    n, emean, eM2 = res
    esigma = np.sqrt(eM2/(n-1))

    assert_array_almost_equal(emean, np.ones((10, 10)), decimal=1)
    assert_almost_equal(esigma, np.zeros((10, 10)), decimal=1)
