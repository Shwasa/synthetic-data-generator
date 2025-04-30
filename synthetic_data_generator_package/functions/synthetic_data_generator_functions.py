import numpy as np
import math
from scipy.spatial import KDTree


# ORIGINAL SYNTHETIC DATA GENERATOR WITH RANDOM DIRECTION (NO COVARIANCE) (NO RESAMPLING)
def original_synth_generator(X, M=None, sigma=0.1, sequential=True, seed=None):
    """
    Generate synthetic data based on X

    Parameters
    ----------
    X : array_like, shape (N, D)
        Array of data to mimic.  Rows are D-dimensional observations.

    M : int
        Number of synthetic data to generate. Default N.

    sigma: float
        Standard deviation of $log(gamma)$ factors.

    sequential: bool | True
        Parents are selected through the data rather than being
        randomly sampled.

    seed: int | None
        Random number generator seed.  None means seed from the time.

    Returns
    -------
    Y : ndarray, shape (M, D)
        Synthetic data
    """
    N, D = X.shape

    if not M: M = N

    mu = math.log((N-1)/M)/D      # Mean of log gamma

    rng = np.random.default_rng(seed)

    Xtree = KDTree(X)
    r = Xtree.query(X, k=2, p=2)[0][:,1]  # Distances to each X's nearest neighbour

    Y = np.zeros((M,D))
    todo = list(range(M))          # Indices of the points still to generate

    # Define the parents in advance
    parent = np.zeros(M, 'int')    # parent[m] is the index of the parent of Y[m]
    copies = int(M/N)
    if sequential:
        parent = np.tile(range(N), copies)
        parent = np.concatenate((parent, np.random.choice(range(N), size=M%N, replace=False)))
    else:
        parent = rng.integers(0, N, size=M)
        
    for m in todo:
        i = parent[m]

        # Create random direction vector
        n = rng.multivariate_normal(np.zeros(D), np.eye(D))
        n /= np.linalg.norm(n)

        # Calculate gamma and create synthetic point
        gamma = math.exp(rng.normal(loc=mu, scale=sigma))
        y = X[i] + gamma*r[i]*n

        Y[m,:] = y
        parent[m] = i
        
    return np.array(parent), np.array(Y)




def fancy_smote(X, sample_size, sigma, everypoint=False, seed=None):
    if seed == None:
        seed = random.randint(0,10000)

    x_list = []
    synth_ds = []
    kdtree = KDTree(X)
    rng = np.random.default_rng(seed)

    # Constants
    N = X.shape[0]
    M = sample_size
    d = X.shape[1]

    i = -1

    for s in range(sample_size):
        # Calculate gamma
        log_gamma = rng.normal((1/d)*(math.log(((N-1)/M))), sigma)
        gamma = np.exp(log_gamma)
                   
        # Create random unit vector
        rand_vec = rng.multivariate_normal(np.zeros(d), np.eye(d))
        rand_unit_vec = rand_vec/np.linalg.norm(rand_vec)

        if everypoint==True:
            # Pick next point in dataset
            i = i + 1
            if i > len(X)-1:
                i = 0     # reset once end is reached
        else:
            # Pick random point from dataset
            i = rng.integers(0, len(X))

        # Pick the point and create parent list
        x = X[i]
        x_list.append(x)

        # Pick its nearest neighbour
        k = 1
        dist,point = kdtree.query(x,k+1) # find the nearest point
        point = point[k]  # choose 2nd nearest point because first nearest point is itself
        dist = dist[k]
        
        # Apply formula
        y = x + gamma*dist*rand_unit_vec
    
        synth_ds.append(y)
    return np.array(x_list), np.array(synth_ds)



#  [BASIC] SYNTHETIC DATA GENERATOR WITH COVARIANCE MATRIX DIRECTION (NO RESAMPLING)
def basic_synth_generator(X, M=None, sigma=0.1, kneighbourhood=20, sequential=True, seed=None):
    """
    Generate synthetic data based on X

    Parameters
    ----------
    X : array_like, shape (N, D)
        Array of data to mimic.  Rows are D-dimensional observations.

    M : int
        Number of synthetic data to generate. Default N.

    sigma: float
        Standard deviation of $log(gamma)$ factors.

    kneighbourhood: int
        Number of neighbours from which to calculate the covariance matrix
        which is used to sample the new direction.

    sequential: bool | True
        Parents are selected through the data rather than being
        randomly sampled.

    seed: int | None
        Random number generator seed.  None means seed from the time.

    Returns
    -------
    Y : ndarray, shape (M, D)
        Synthetic data
    """
    N, D = X.shape

    if not M: M = N

    mu = math.log((N-1)/M)/D      # Mean of log gamma

    rng = np.random.default_rng(seed)

    Xtree = KDTree(X)
    r = Xtree.query(X, k=2, p=2)[0][:,1]  # Distances to each X's nearest neighbour

    Y = np.zeros((M,D))
    todo = list(range(M))          # Indices of the points still to generate

    # Define the parents in advance
    parent = np.zeros(M, 'int')    # parent[m] is the index of the parent of Y[m]
    copies = int(M/N)
    if sequential:
        parent = np.tile(range(N), copies)
        parent = np.concatenate((parent, np.random.choice(range(N), size=M%N, replace=False)))
    else:
        parent = rng.integers(0, N, size=M)
        
    for m in todo:
        i = parent[m]

        # Get the local covariance matrix
        _, Ineighbours = Xtree.query(X[i], k=kneighbourhood, p=2)
        S = np.cov(X[Ineighbours].T)

        # Create direction vector using local covariance
        n = rng.multivariate_normal(np.zeros(D), S)
        n /= np.linalg.norm(n)

        # Calculate gamma and create synthetic point
        gamma = math.exp(rng.normal(loc=mu, scale=sigma))
        y = X[i] + gamma*r[i]*n

        Y[m,:] = y
        parent[m] = i
        
    return np.array(parent), np.array(Y)



# [RESAMPLING] SYNTHETIC DATA GENERATOR WITH RESAMPLING
def resampling_synth_generator(X, M=None, sigma=0.1, kneighbourhood=20, resamples=1, repeats=1, sequential=True, tol=0.99, seed=None, verbose=False):
    """
    Generate synthetic data based on X

    Parameters
    ----------
    X : array_like, shape (N, D)
        Array of data to mimic.  Rows are D-dimensional observations.

    M : int
        Number of synthetic data to generate. Default N.

    sigma: float
        Standard deviation of $log(gamma)$ factors.

    kneighbourhood: int
        Number of neighbours from which to calculate the covariance matrix
        which is used to sample the new direction.

    resamples: int
        Number of times to resample the direction to avoid collisions
        with the original data.

    repeats: int
        Number of times to repeat sampling to avoid collisions.

    sequential: bool | True
        Parents are selected through the data rather than being
        randomly sampled.

    tol: float (0 < tol < 1):
        Tolerance on how close the synthetic neighbour can be to
        the nearest neighour in the original data. Default: 0.99.

    seed: int | None
        Random number generator seed.  None means seed from the time.

    verbose: bool
        Print information about the convergence

    Returns
    -------
    Y : ndarray, shape (M, D)
        Synthetic data
    """
    N, D = X.shape

    if not M: M = N

    mu = math.log((N-1)/M)/D      # Mean of log gamma

    rng = np.random.default_rng(seed)

    Xtree = KDTree(X)
    r = Xtree.query(X, k=2, p=2)[0][:,1]  # Distances to each X's nearest neighbour

    Y = np.zeros((M,D))
    todo = list(range(M))          # Indices of the points still to generate

    # Define the parents in advance
    parent = np.zeros(M, 'int')    # parent[m] is the index of the parent of Y[m]
    copies = int(M/N)
    if sequential:
        parent = np.tile(range(N), copies)
        parent = np.concatenate((parent, np.random.choice(range(N), size=M%N, replace=False)))
    else:
        parent = rng.integers(0, N, size=M)
        
    for repeat in range(repeats):
        for m in todo:
            i = parent[m]

            # Get the local covariance matrix
            _, Ineighbours = Xtree.query(X[i], k=kneighbourhood, p=2)
            S = np.cov(X[Ineighbours].T)

            # Sample for the new point until one that is not too close is found
            # If this doesn't work after resamples tries, we just use the last attempt
            for _ in range(resamples):
                n = rng.multivariate_normal(np.zeros(D), S)
                n /= np.linalg.norm(n)
                gamma = math.exp(rng.normal(loc=mu, scale=sigma))
                y = X[i] + gamma*r[i]*n
                # Check it's actually not too close to something else in X
                s = Xtree.query(y, k=1, p=2)[0]
                if s >= tol*r[i]*gamma:
                    break

            Y[m,:] = y
            parent[m] = i

        Ytree = KDTree(Y)
        s, yneighbour = Ytree.query(X, k=1, p=2)
        # s/r < 1 means that a point in X is closer to a point in Y
        # than its nearest neighbour in X.
        tooclose = s < tol*r
        todo = yneighbour[tooclose]
        if verbose:
            print('Repeat', repeat, end=' ')
            print(f'Todo: {len(todo)}', end='  ')
            print(f'Closest: {min(s):.3g}')
        if len(todo) == 0:
            break
        
    return np.array(parent), np.array(Y)