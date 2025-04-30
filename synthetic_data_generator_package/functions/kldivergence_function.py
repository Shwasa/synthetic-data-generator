import numpy as np
from scipy.spatial import cKDTree as KDTree

def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    x : array (n,d)
        Samples from distribution P, which typically represents the true distribution.
    y : array (m,d)
        Samples from distribution Q, which typically represents the approximate distribution.

    If x and y are one-dimensional arrays, they are converted to two-dimensions.

    Returns
    -------
    D : float
         The estimated Kullback-Leibler divergence D(P||Q).

    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of continuous distributions
    IEEE International Symposium on Information Theory, 2008.

    Adapted from https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
    """
    if len(x.shape) == 1:
        x = x[:,np.newaxis] # if 1 dimensional converts array to 2 dimensional
    if len(y.shape) == 1:
        y = y[:,np.newaxis]

    n, d = x.shape
    m, dy = y.shape
    assert d == dy

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.  For y, use the second nearest neighbour if the
    # distance to the first is zero. This allows KLdivergence(x, x) to be calculated.
    r = xtree.query(x, k=2, p=2)[0][:,1]
    s = ytree.query(x, k=2, p=2)[0]
    s = np.where(s[:,0] == 0.0, s[:,1], s[:,0])
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m/(n - 1))

def normalKL(Pmu, Psigma, Qmu, Qsigma):
    """
    Return the KL divergence between one-dimensional normal distributions.

    Parameters
    ----------
    Pmu : scalar
        Mean of P
    Psigma : scalar
        Standard deviation of P
    Qmu : scalar
        Mean of Q
    Qsigma : scalar
        Standard deviation of Q

    Returns
    -------
    Dkl : scalar

        The KL divergence KL[P||Q]
    """
    Dkl = (Psigma/Qsigma)**2 + ((Qmu-Pmu)/Qsigma)**2 - 1 + 2*np.log(Qsigma/Psigma)
    Dkl /= 2
    return Dkl


def JensenShannonDivergence(x, y):
    """
    Estimate the Jensen-Shannon distance JSD(p, q) from samples x of p and y of q
    
    Parameters
    ----------
    x : array_like
        Input array containing samples of p
    y : array_like
        Input array containing samples of q
    
    If x and y are one-dimensional arrays, they are converted to two-dimensions.

    Returns
    -------
    
    JSD : scalar
        Estimate of the Jensen Shannon divergence in nats. 
    """
    if len(x.shape) == 1:
        x = x[:,np.newaxis]
    if len(y.shape) == 1:
        y = y[:,np.newaxis]

    # Sample from the larger set to get an equal number of samples
    # from each to form the mixture
    if x.shape[0] < y.shape[0]:
        short, long = x, y
    else:
        short, long = y, x
    if long.shape[0] > short.shape[0]:
        np.random.shuffle(long)
        long = long[:short.shape[0],:]
        
    xm = np.vstack((long, short))  # Samples from mixture distribution
    
    d = KLdivergence(x, xm) + KLdivergence(y, xm)
    return d/2


if __name__ == "__main__":
    # Minimal tests/demonstation.
    import matplotlib.pylab as plt
    N = 10000
    M = 20000

    Mu = np.linspace(-5, 5, 11)
    dklpq = []
    dklqp = []
    tklpq = []
    tklqp = []
    jsd = []
    p = np.random.randn(N)
    qstd = 2
    for mu in Mu:
        q = qstd*np.random.randn(M) + mu

        dklpq.append(KLdivergence(p, q))
        tklpq.append(normalKL(0, 1, mu, qstd))

        dklqp.append(KLdivergence(q, p))
        tklqp.append(normalKL(mu, qstd, 0, 1))

        jsd.append(JensenShannonDivergence(p, q))

    plt.plot(Mu, dklpq, '-o', c='tab:blue', label='Estimated D(p||q)')
    plt.plot(Mu, tklpq, '--s', c='tab:blue', alpha=0.5, label='True D(p||q)')
    plt.plot(Mu, dklqp, '-o', c='tab:orange', label='Estimated D(q||p)')
    plt.plot(Mu, tklqp, '--s', c='tab:orange', alpha=0.5, label='True D(q||p)')
    plt.plot(Mu, jsd, '-o', c='tab:red', label='Estimated JSD(q||p)')
    plt.title(f'KL divergence   N = {N}  M = {M}')
    plt.xlabel('Difference in means')
    plt.legend()
    plt.grid()
    plt.show()
