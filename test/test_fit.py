import pytest
import numpy
from matplotlib import pyplot
from rrblup.fit import rrBLUP_REML_calc_S, rrBLUP_REML_calc_SZZtS, rrBLUP_REML_calc_lamb_Ur, rrBLUP_REML_calc_nusq, rrBLUP_REML_logLik, rrBLUP_REML_logLik_deriv, rrBLUP_REML_grid_search, rrblup_REML_NR_search

# numpy.random.seed(18127648)

@pytest.fixture
def nobs():
    yield 10

@pytest.fixture
def ntrait():
    yield 2

@pytest.fixture
def nfixed():
    yield 3

@pytest.fixture
def nmkr():
    yield 100

@pytest.fixture
def Xmat(nobs, nfixed):
    yield numpy.random.random(size = (nobs, nfixed))

@pytest.fixture
def Bmat(nfixed, ntrait):
    yield numpy.random.random(size = (nfixed, ntrait))

@pytest.fixture
def Zmat(nobs, nmkr):
    # generate {0,1,2} genotype matrix
    out = numpy.random.binomial(2, 0.5, size = (nobs, nmkr))
    # convert to {-1,0,1} format
    out -= 1
    yield out
    # yield numpy.random.normal(size = (nobs, nmkr))

@pytest.fixture
def Umat(nmkr, ntrait):
    yield numpy.random.random(size = (nmkr, ntrait))

@pytest.fixture
def Emat(Zmat, Umat, nobs, ntrait):
    SDmat = 0.5 * numpy.std(Zmat @ Umat, axis = 0)
    yield numpy.random.normal(scale = SDmat, size = (nobs, ntrait))
    # yield 0

@pytest.fixture
def Ymat(Xmat, Bmat, Zmat, Umat, Emat):
    yield Xmat @ Bmat + Zmat @ Umat + Emat

### Tests ###

def test_rrBLUP_REML_calc_S(Xmat, nobs):
    # calculate observed and expected S matrices
    Sobs = rrBLUP_REML_calc_S(Xmat)
    Sexp = numpy.identity(nobs) - (Xmat @ numpy.linalg.inv(Xmat.T @ Xmat) @ Xmat.T)
    # assert correct shapes
    assert Sobs.ndim == 2
    assert Sobs.shape == (nobs,nobs)
    # assert that the difference between observed and expected values are small
    diff = numpy.sum(numpy.absolute(Sobs - Sexp))
    assert diff < 1e-10
    # assert that matrix is symmetric
    diff = numpy.sum(numpy.absolute(Sobs - Sobs.T))
    assert diff < 1e-10
    # assert that matrix is idempotent
    diff = numpy.sum(numpy.absolute(Sobs @ Sobs - Sobs))
    assert diff < 1e-10

def test_rrBLUP_REML_calc_SZZtS(Xmat, Zmat, nobs):
    # calculate S matrix
    Smat = rrBLUP_REML_calc_S(Xmat)
    # calculate observed and expected matrices
    Mobs = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
    Mexp = Smat @ Zmat @ Zmat.T @ Smat
    # assert correct shapes
    assert Mobs.ndim == 2
    assert Mobs.shape == (nobs,nobs)
    # assert that the difference between observed and expected values are small
    diff = numpy.sum(numpy.absolute(Mobs - Mexp))
    assert diff < 1e-10
    # assert that matrix is symmetric
    diff = numpy.sum(numpy.absolute(Mobs - Mobs.T))
    assert diff < 1e-10

def test_rrBLUP_REML_calc_lamb_Ur(Xmat, Zmat, nobs, nfixed):
    # calculate S and SHS matrices
    Smat = rrBLUP_REML_calc_S(Xmat)
    SHSmat = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
    # calculate eigenvalues and eigenvectors
    lamb, Ur = rrBLUP_REML_calc_lamb_Ur(SHSmat, nfixed)
    # assert correct shapes
    assert lamb.ndim == 1
    assert Ur.ndim == 2
    assert lamb.shape == (nobs-nfixed,)
    assert Ur.shape == (nobs-nfixed,nobs)
    # assert that the eigenvalues are all positive and non-zero
    assert numpy.all(lamb > 0.0)
    # assert that the eigenvectors are orthogonal to each other
    diff = numpy.sum(numpy.absolute(numpy.identity(nobs-nfixed) - Ur @ Ur.T))
    assert diff < 1e-10

def test_rrBLUP_REML_calc_nusq(Xmat, Zmat, Ymat, nobs, nfixed, ntrait):
    # calculate S, SHS, eigenvalue, eigenvector matrices
    Smat = rrBLUP_REML_calc_S(Xmat)
    SHSmat = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
    lambvec, Urmat = rrBLUP_REML_calc_lamb_Ur(SHSmat, nfixed)
    # calculate observed and expected matrices
    NuSqobs = rrBLUP_REML_calc_nusq(Urmat, Ymat)
    NuSqexp = (Urmat @ Ymat)**2
    # assert correct shapes
    assert NuSqobs.ndim == 2
    assert NuSqobs.shape == (nobs-nfixed,ntrait)
    # assert that the difference between observed and expected values is small
    diff = numpy.sum(numpy.absolute(NuSqobs - NuSqexp))
    assert diff < 1e-10

def test_rrBLUP_REML_logLik(Xmat, Zmat, Ymat, nobs, nfixed, ntrait):
    # calculate S, SHS, eigenvalue, eigenvector, nu matrices
    Smat = rrBLUP_REML_calc_S(Xmat)
    SHSmat = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
    lambvec, Urmat = rrBLUP_REML_calc_lamb_Ur(SHSmat, nfixed)
    NuSqmat = rrBLUP_REML_calc_nusq(Urmat, Ymat)
    # construct grid of values to test delta values
    grid = numpy.logspace(numpy.log(1e-5), numpy.log(1e5), 100, base=numpy.e)
    # test grid of values for log-likelihood function
    for value in grid:
        for trait in range(ntrait):
            ll = rrBLUP_REML_logLik(value, nobs, nfixed, lambvec, NuSqmat[:,trait])
            assert numpy.isfinite(ll)   # assert value is not NaN, -Inf, or Inf
            assert numpy.exp(ll) > 0    # assert likelihood is > 0
    # test cases where delta <<< 0; these scenarios are impossible
    ll = rrBLUP_REML_logLik(-1e5, nobs, nfixed, lambvec, NuSqmat[:,0])
    assert not numpy.isfinite(ll)
    ll = rrBLUP_REML_logLik(-1e5, nobs, nfixed, lambvec, NuSqmat[:,1])
    assert not numpy.isfinite(ll)
    # plot values in grid
    grid = numpy.logspace(numpy.log(1e-5), numpy.log(1e5), 1000, base=numpy.e)
    Y = numpy.empty((ntrait, len(grid)))
    for i in range(ntrait):
        for j,value in enumerate(grid):
            Y[i,j] = rrBLUP_REML_logLik(value, nobs, nfixed, lambvec, NuSqmat[:,i])
    for trait in range(ntrait):
        pyplot.plot(numpy.log(grid), Y[trait], label = "Trait{0}".format(trait))
    pyplot.xlabel("log(delta)")
    pyplot.ylabel("log-likelihood")
    pyplot.legend()
    pyplot.savefig("rrBLUP_REML_logLik.png")
    pyplot.close()


def test_rrBLUP_REML_logLik_deriv(Xmat, Zmat, Ymat, nobs, nfixed, ntrait):
    # calculate S, SHS, eigenvalue, eigenvector, nu matrices
    Smat = rrBLUP_REML_calc_S(Xmat)
    SHSmat = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
    lambvec, Urmat = rrBLUP_REML_calc_lamb_Ur(SHSmat, nfixed)
    NuSqmat = rrBLUP_REML_calc_nusq(Urmat, Ymat)
    # construct grid of values to test delta values
    grid = numpy.logspace(numpy.log(1e-5), numpy.log(1e5), 100, base=numpy.e)
    # test grid of values for log-likelihood function
    for value in grid:
        for trait in range(ntrait):
            dll = rrBLUP_REML_logLik_deriv(value, nobs, nfixed, lambvec, NuSqmat[:,trait])
            assert numpy.isfinite(dll)  # assert value is not NaN, -Inf, or Inf
            assert numpy.exp(dll) > 0   # assert likelihood is > 0
    # test cases where delta <<< 0; these scenarios are impossible
    dll = rrBLUP_REML_logLik(-1e5, nobs, nfixed, lambvec, NuSqmat[:,0])
    assert not numpy.isfinite(dll)
    dll = rrBLUP_REML_logLik(-1e5, nobs, nfixed, lambvec, NuSqmat[:,1])
    assert not numpy.isfinite(dll)
    # plot values in grid
    grid = numpy.logspace(numpy.log(1e-5), numpy.log(1e5), 1000, base=numpy.e)
    Y = numpy.empty((ntrait, len(grid)))
    for i in range(ntrait):
        for j,value in enumerate(grid):
            Y[i,j] = rrBLUP_REML_logLik_deriv(value, nobs, nfixed, lambvec, NuSqmat[:,i])
    for trait in range(ntrait):
        pyplot.plot(numpy.log(grid), Y[trait], label = "Trait{0}".format(trait))
    pyplot.xlabel("log(delta)")
    pyplot.ylabel("log-likelihood derivative")
    pyplot.legend()
    pyplot.savefig("rrBLUP_REML_logLik_deriv.png")
    pyplot.close()

# def test_rrBLUP_REML_NR_search(Xmat, Zmat, Ymat, nobs, nfixed, ntrait):
#     # calculate S, SHS, eigenvalue, eigenvector, nu matrices
#     Smat = rrBLUP_REML_calc_S(Xmat)
#     SHSmat = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
#     lambvec, Urmat = rrBLUP_REML_calc_lamb_Ur(SHSmat, nfixed)
#     NuSqmat = rrBLUP_REML_calc_nusq(Urmat, Ymat)
#     # test output
#     out = rrblup_REML_NR_search(nobs, nfixed, lambvec, NuSqmat[:,0], maxiter=1e3)
#     assert isinstance(out, tuple)
#     print(out)
#     assert False

# def test_rrBLUP_REML_grid_search(Xmat, Zmat, Ymat, nobs, nfixed, ntrait):
#     # calculate S, SHS, eigenvalue, eigenvector, nu matrices
#     Smat = rrBLUP_REML_calc_S(Xmat)
#     SHSmat = rrBLUP_REML_calc_SZZtS(Smat, Zmat)
#     lambvec, Urmat = rrBLUP_REML_calc_lamb_Ur(SHSmat, nfixed)
#     NuSqmat = rrBLUP_REML_calc_nusq(Urmat, Ymat)
#     # test output
#     out = rrBLUP_REML_grid_search(nobs, nfixed, lambvec, NuSqmat[:,0])
#     assert isinstance(out, tuple)
