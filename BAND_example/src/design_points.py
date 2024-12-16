import numpy as np


def detmax(A, n, max_iter=1000, tol=1e-6):
    N = A.shape[0]

    # initialize
    initidx = np.random.choice(N, size=n, replace=False)

    inidx = initidx.copy()
    outidx = np.setdiff1d(np.arange(N), inidx)

    for iter in range(max_iter):        
        B = A[inidx]
        W, U = np.linalg.eigh(B.T @ B)
        Bh = U / np.sqrt(W) @ U.T
        
        logdet = np.log(W).sum()
        plus2  = ((A[outidx] @ Bh)**2).sum(1)
        minus2 = ((A[inidx] @ Bh)**2).sum(1)
        pm2 = (A[inidx] @ Bh @ Bh.T @ A[outidx].T) ** 2

        delta = pm2 - np.outer(minus2, 1 + plus2) + plus2

        maxdelta_ind = np.argmax(delta)
        maxdelta_loc = (maxdelta_ind // (N - n), maxdelta_ind % (N - n))

        if delta[maxdelta_loc] < tol:
            print('maxdelta = {:.3E}, falling below tolerance of {:.3E} at '\
                  'iteration {:d}'.format(delta[maxdelta_loc], tol, iter))
            break
        else:
            whichout = outidx[maxdelta_loc[1]]
            whichin = inidx[maxdelta_loc[0]]
            
            # Swap the indices
            inidx[maxdelta_loc[0]] = whichout
            outidx[maxdelta_loc[1]] = whichin

    if iter >= max_iter - 1:
        print('maxdelta = {:.3E}, reached iteration {:d}'.format(delta[maxdelta_loc], iter))

    B = A[inidx]
    return B, initidx, inidx

def load_data(train_size, validation_size, priors, seed, dmax):
    
    def latin_hypercube_sampling(dimensions, samples, seed):
        np.random.seed(seed)
        result = np.empty((samples, dimensions))
        for i in range(dimensions):
            points = np.linspace(0, 1, samples, endpoint=False) + np.random.rand(samples) / samples
            np.random.shuffle(points)
            result[:, i] = points

        return result

    n_samples = train_size + validation_size

    lhs_samples = latin_hypercube_sampling(dimensions=len(priors), samples=n_samples, seed=seed)

    scaled_samples = np.zeros_like(lhs_samples)
    for i, key in enumerate(priors.keys()):
        xmin = priors[key].minimum
        xmax = priors[key].maximum
        scaled_samples[:, i] = xmin + lhs_samples[:, i] * (xmax - xmin)


    if dmax == True:
        _, initidx, inidx = detmax(scaled_samples, train_size)
        print('detmax')
    else: 
        np.random.seed(seed)
        inidx = np.random.choice(range(len(scaled_samples)), size=train_size, replace=False)
    
    train_indices = np.array(inidx)
    remaining_indices = np.setdiff1d(np.arange(len(scaled_samples)), train_indices)
    validation_indices = remaining_indices[:validation_size]
    
    train_points = scaled_samples[train_indices]
    validation_points = scaled_samples[validation_indices]

    return train_points, validation_points

if __name__ == "__main__":
    train_points, validation_points = load_data(train_size=30, validation_size=20, seed=42)
