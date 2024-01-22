from numbers import Real
import numpy
import pandas
import pytest
from matplotlib import pyplot
from rrblup.fitML0 import rrBLUP_ML0
from rrblup.fitML0 import rrBLUP_ML0_calc_G
from rrblup.fitML0 import rrBLUP_ML0_calc_d_V
from rrblup.fitML0 import rrBLUP_ML0_calc_etasq
from rrblup.fitML0 import rrBLUP_ML0_center_y
from rrblup.fitML0 import rrBLUP_ML0_neg2LogLik_fast
from rrblup.fitML0 import rrBLUP_ML0_nonzero_d_V
from rrblup.fitML0 import rrBLUP_ML0_calc_ridge
from rrblup.fitML0 import rrBLUP_ML0_calc_ZtZplI
from rrblup.fitML0 import rrBLUP_ML0_calc_Zty
from rrblup.fitML0 import gauss_seidel

@pytest.fixture
def yvec():
    traits_df = pandas.read_csv("traits.csv")
    out = traits_df.iloc[:,1].to_numpy(dtype = float)
    yield out

@pytest.fixture
def Zmat():
    markers_df = pandas.read_csv("markers.csv")
    out = markers_df.to_numpy(dtype = float)
    yield out

@pytest.fixture
def nobs(yvec):
    yield len(yvec)

@pytest.fixture
def nmkr(Zmat):
    yield Zmat.shape[1]

@pytest.fixture
def varE():
    yield 0.5725375

@pytest.fixture
def varU():
    yield 0.4690475

#############
### Tests ###
#############

def test_rrBLUP_ML0_calc_G(Zmat, nobs):
    G = rrBLUP_ML0_calc_G(Zmat)
    assert isinstance(G, numpy.ndarray)
    assert G.shape == (nobs,nobs)

def test_rrBLUP_ML0_center_y(yvec, nobs):
    y = rrBLUP_ML0_center_y(yvec)
    assert isinstance(y, numpy.ndarray)
    assert y.shape == (nobs,)
    assert abs(y.mean(0)) < 1e-10

def test_rrBLUP_ML0_calc_d_V(Zmat, nobs):
    G = rrBLUP_ML0_calc_G(Zmat)
    out = rrBLUP_ML0_calc_d_V(G)
    assert isinstance(out, tuple)
    assert len(out) == 2
    d = out[0]
    assert isinstance(d, numpy.ndarray)
    assert d.shape == (nobs,)
    V = out[1]
    assert isinstance(V, numpy.ndarray)
    assert V.shape == (nobs,nobs)

def test_rrBLUP_ML0_nonzero_d_V(Zmat, nobs):
    G = rrBLUP_ML0_calc_G(Zmat)
    d, V = rrBLUP_ML0_calc_d_V(G)
    out = rrBLUP_ML0_nonzero_d_V(d, V)
    assert isinstance(out, tuple)
    assert len(out) == 2
    d = out[0]
    assert isinstance(d, numpy.ndarray)
    assert d.shape == (nobs-1,)
    V = out[1]
    assert isinstance(V, numpy.ndarray)
    assert V.shape == (nobs,nobs-1)

def test_rrBLUP_ML0_calc_etasq(yvec, Zmat, nobs):
    G = rrBLUP_ML0_calc_G(Zmat)
    d, V = rrBLUP_ML0_calc_d_V(G)
    d, V = rrBLUP_ML0_nonzero_d_V(d, V)
    y = rrBLUP_ML0_center_y(yvec)
    etasq = rrBLUP_ML0_calc_etasq(y, V)
    assert isinstance(etasq, numpy.ndarray)
    assert etasq.shape == (nobs-1,)

def test_rrBLUP_ML0_neg2logLik_fast(yvec, Zmat, nobs, varE, varU):
    G = rrBLUP_ML0_calc_G(Zmat)
    d, V = rrBLUP_ML0_calc_d_V(G)
    d, V = rrBLUP_ML0_nonzero_d_V(d, V)
    y = rrBLUP_ML0_center_y(yvec)
    etasq = rrBLUP_ML0_calc_etasq(y, V)
    out = rrBLUP_ML0_neg2LogLik_fast(numpy.log([varE, varU]), etasq, d, nobs)
    assert isinstance(out, Real)
    # plot contour of function for visual examination
    pts = numpy.linspace(-1, 0, 30)
    gridpts = numpy.meshgrid(pts, pts)
    gridX = gridpts[0] # (g,g) containing log(varE) values
    gridY = gridpts[1] # (g,g) containing log(varU) values
    gridZ = numpy.empty(gridX.shape, dtype = float)
    for i in range(gridX.shape[0]):
        for j in range(gridX.shape[1]):
            gridZ[i,j] = rrBLUP_ML0_neg2LogLik_fast((gridX[i,j],gridY[i,j]), etasq, d, len(y))
    fig, ax = pyplot.subplots()
    CS = ax.contour(gridX, gridY, gridZ, levels = 10)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel("log(varE)")
    ax.set_ylabel("log(varU)")
    ax.set_title('-2 * log-likelihood (minimizing)')
    pyplot.savefig("neg2LogLik.png")
    pyplot.close()
    
def test_rrBLUP_ML0_calc_ridge(varE, varU):
    ridge = rrBLUP_ML0_calc_ridge(varE, varU)
    assert isinstance(ridge, Real)
    assert ridge == (varE / varU)

def test_rrBLUP_ML0_calc_ZtZplI(Zmat, varE, varU, nmkr):
    ridge = rrBLUP_ML0_calc_ridge(varE, varU)
    A = rrBLUP_ML0_calc_ZtZplI(Zmat, ridge)
    assert isinstance(A, numpy.ndarray)
    assert A.shape == (nmkr,nmkr)

def test_rrBLUP_ML0_calc_Zty(Zmat, yvec, nmkr):
    y = rrBLUP_ML0_center_y(yvec)
    b = rrBLUP_ML0_calc_Zty(Zmat, y)
    assert isinstance(b, numpy.ndarray)
    assert b.shape == (nmkr,)

def test_gauss_seidel(yvec, Zmat, nmkr, varE, varU):
    ridge = rrBLUP_ML0_calc_ridge(varE, varU)
    A = rrBLUP_ML0_calc_ZtZplI(Zmat, ridge)
    y = rrBLUP_ML0_center_y(yvec)
    b = rrBLUP_ML0_calc_Zty(Zmat, y)
    x = gauss_seidel(A, b)
    assert isinstance(x, numpy.ndarray)
    assert x.shape == (nmkr,)

def test_rrBLUP_ML0(yvec, Zmat):
    out = rrBLUP_ML0(yvec, Zmat)
    assert isinstance(out, dict)

# code for running in terminal
if False:
    import pandas
    from rrblup.fitML0 import rrBLUP_ML0
    traits_df = pandas.read_csv("traits.csv")
    y = traits_df.iloc[:,1].to_numpy(dtype = float)
    markers_df = pandas.read_csv("markers.csv")
    Z = markers_df.to_numpy(dtype = float)
    fmML0 = rrBLUP_ML0(y, Z)
    yhat = fmML0["yhat"]
    1.0 - ((y - yhat)**2).sum() / (y**2).sum()
