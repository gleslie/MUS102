"""
tonality.py - Matrix representations of musical scores, corpara, and their tonality

Example: J. S. Bach's "Well Tempered Clavier" Books 1 and 2

2017, Michael A. Casey, Dartmouth College, Bregman Media Labs

License: 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
http://creativecommons.org/licenses/by-nc/4.0/
"""

from pylab import *
import glob

pc_labels = tile(['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'],13)

def load_wtc(idx=None, win_len=1, sample_len=0):
    """
    Load scores in matrix form in the entire WTC dataset.
    Inputs:
            idx - slice argument giving range of works [None] (all)
        win_len - num tactus beats to integrate [1] (no integration)
     sample_len - number of sampled windows per work [0] (all)
    """
    flist = sorted(glob.glob('*.ascii'))
    if idx is not None:
        if not iterable(idx):
            idx = [idx]
    else:
        idx = range(len(flist))
    flist = array(flist)[idx]
    if win_len>0:
        A = [win_mtx(loadtxt(fname, dtype='i4'),win_len) for fname in flist]
    else:
        A = [loadtxt(fname, dtype='i4').mean(1) for fname in flist]
    if win_len>0 and sample_len>0:
        AA = [a[:,random.permutation(a.shape[1])[:sample_len]] for a in A]
    else:
        AA = A
    return AA

def euc_normed(A,B):
    """
    d = euc_normed(A,B)
        Return the normed Euclidean distance between two matrices
        Second dimension (num columns) of A and B must be the same
    """
    nz_idx = (A*A).sum(1) > 0
    AA = zeros(A.shape)
    BB = zeros(B.shape)    
    AA[nz_idx] = (A[nz_idx].T / sqrt((A[nz_idx]*A[nz_idx]).sum(1))).T
    nz_idx = (B*B).sum(1) > 0    
    BB[nz_idx] = (B[nz_idx].T / sqrt((B[nz_idx]*B[nz_idx]).sum(1))).T
    D = 2 - 2 * dot(AA,BB.T)
    D[where(D<finfo(float32).eps)]=0
    return sqrt(D)

def win_mtx(a, win_len=2):
    """
    Options:
        win_len  - window length [2]
    """
    # perform simple integration
    N = ceil(a.shape[1]/float(win_len))
    aa = []
    for k in arange(N-1):
        aa.append(a[:,k*win_len:(k+1)*win_len].mean(1)) 
    return vstack(aa).T

def fold_mtx(a):
    """
    Fold piano-roll matrix into single octave beginning with 'C'.
    """
    return a[:120,:].reshape(-1,12,a.shape[1]).mean(0)

def dissimilarity_mtx(A):
    """
    Given a piano-roll indicator matrix, construct self-dissimilarity matrix
    """
    D = euc_normed(A.T,A.T)
    return D

def center_mtx(D):
    """
    Given a dissimilarity or dissonance matrix, center the matrix by subtracting the mean of 
    the rows and columns. For a dissimilarity matrix this operation yields the "scatter matrix".
    """
    H = eye(D.shape[0]) - 1.0/D.shape[0]
    B = dot(dot(H,-0.5*D),H)
    return B

def dissonance_mtx(A):
    """
    Given a piano-roll indicator matrix, construct pair-wise dissonance matrix    
    """
    n = A.shape[1]
    D = zeros((n,n))
    for i,a in enumerate(A.T[:-1]):
        for j,b in enumerate(A.T[i+1:]):
            D[i,j] = diss_fun(expand_dims(a+b,1))

def dissonance_fun(A):
    """
    Given a piano-roll indicator matrix representation of a musical work (128 pitches x beats),
    return the dissonance as a function of beats.
    Input:
        A  - 128 x beats indicator matrix of MIDI pitch number

    """
    freq_rats = arange(1,11) # Harmonic series ratios
    amps = exp(-.5 * freq_rats) # Partial amplitudes
    F0 = 8.1757989156 # base frequency for MIDI (note 0)
    diss = [] # List for dissonance values
    thresh = 1e-3
    for beat in A.T:
        idx = where(beat>thresh)[0]
        if len(idx):
            freqs, mags = [], [] # lists for frequencies, mags
            for i in idx:
                freqs.extend(F0*2**(i/12.0)*freq_rats)
                mags.extend(amps)
            freqs = array(freqs)
            mags = array(mags)
            sortIdx = freqs.argsort()
            d = _dissonance_fun(freqs[sortIdx],mags[sortIdx])
            diss.extend([d])
        else:
            diss.extend([-1]) # Null value
    return array(diss)


def _dissonance_fun(freqs, amps=None, params=None):
    """
    ::
        Compute dissonance between partials with center frequencies in freqs,
        and amplitudes in amps. Based on William Sethares after Plomp and Levelt:

        default params = (-3.51, -5.75, 0.0207, 19.96, 5, -5, 0.24)
        default amps, use 1 as amplitude for all partials.
    """
    if params == None: params = (-3.51, -5.75, 0.0207, 19.96, 5, -5, 0.24)
    b1, b2, s1, s2, c1, c2, Dstar  = params
    if amps == None: amps = [1]*len(freqs)
    f = array(freqs)
    a = array(amps)
    idx = argsort(f)
    f = f[idx]
    a = a[idx]
    N = f.size
    D = 0
    for i in range(1, N):
        Fmin = f[ 0 : N - i ]
        S = Dstar / ( s1 * Fmin + s2)
        Fdif = f[ i : N ] - f[ 0 : N - i ]
        am = a[ i : N ] * a[ 0 : N - i ]
        Dnew = am * (c1 * exp (b1 * S * Fdif) + c2 * exp(b2 * S * Fdif))
        D += Dnew.sum()
    return D

def plot_mtx(mtx=None, title=None, newfig=False, cbar=True, **kwargs):
    """
    ::

        static method for plotting a matrix as a time-frequency distribution (audio features)
    """
    if mtx is None or type(mtx) != ndarray:
        raise ValueError('First argument, mtx, must be a array')
    if newfig: figure()
    dbscale = kwargs.pop('dbscale', False) 
    bels = kwargs.pop('bels',False)
    norm = kwargs.pop('norm',False)
    normalize = kwargs.pop('normalize',False)
    origin=kwargs.pop('origin','lower')
    aspect=kwargs.pop('aspect','auto')
    interpolation=kwargs.pop('interpolation','nearest')
    cmap=kwargs.pop('cmap',cm.gray_r)
    clip=-100.
    X = scale_mtx(mtx, normalize=normalize, dbscale=dbscale, norm=norm, bels=bels)
    i_min, i_max = where(X.mean(1))[0][[0,-1]]
    X = X[i_min:i_max+1].copy()
    if dbscale or bels:
        if bels: clip/=10.
        imshow(clip(X,clip,0),origin=origin, aspect=aspect, interpolation=interpolation, cmap=cmap, **kwargs)
    else:
        imshow(X,origin=origin, aspect=aspect, interpolation=interpolation, cmap=cmap, **kwargs)
    if title:
        title(title,fontsize=16)
    if cbar:
        colorbar()
    yticks(arange(0,i_max+1-i_min,3),pc_labels[i_min:i_max+1:3],fontsize=14)
    xlabel('Tactus', fontsize=14)
    ylabel('MIDI Pitch', fontsize=14)
    grid()

def scale_mtx(M, normalize=False, dbscale=False, norm=False, bels=False):
    """
    ::

        Perform mutually-orthogonal scaling operations, otherwise return identity:
          normalize [False]
          dbscale  [False]
          norm      [False]        
    """
    if not (normalize or dbscale or norm or bels):
        return M
    else:
        X = M.copy() # don't alter the original
        if norm:
            nz_idx = (X*X).sum(1) > 0
            X[nz_idx] = (X[nz_idx].T / sqrt((X[nz_idx]*X[nz_idx]).sum(1))).T
        if normalize:
            X=X-min(X)
            X=X/max(X)
        if dbscale or bels:
            X = log10(clip(X,0.0001,X.max()))
            if dbscale:                
                X = 20*X
    return X

def hist_mtx(mtx, tstr=''):
    """
    Given a piano-roll matrix, 128 MIDI piches x beats, plot the pitch class histogram
    """
    i_min, i_max = where(mtx.mean(1))[0][[0,-1]]
    stem(arange(i_max+1-i_min),mtx[i_min:i_max+1,:].sum(1))
    ttl = 'Note Frequency'
    if tstr: ttl+=': '+tstr
    title(ttl,fontsize=16)
    t=xticks(arange(0,i_max+1-i_min,3),pc_labels[i_min:i_max+1:3],fontsize=14)
    xlabel('Pitch Class', fontsize=14)
    ylabel('Frequency', fontsize=14)
    ax = axis()
    axis(xmin=-0.5)
    grid()

if __name__ == "__main__":
    interactive(True)
    a = loadtxt('BachWTC1/I_Prelude01.ascii')
    figure()
    # Plot piano roll: MIDI pitch by beats
    subplot(211)
    plot_mtx(a, cmap=cm.gray_r, cbar=False)
    axis('tight')
    title('WTC 1 "Prelude in C": Piano Roll')

    # Plot dissonance by (integrated) beats
    subplot(212)
    win_len=8 # Number of beats to integrate, non-overlapping
    a = win_mtx(a, win_len)
    d = dissonance_fun(a)
    plot(arange(len(d))*win_len, d,'r',linewidth=1)
    axis('tight')
    title('Dissonance (win_len=%d)'%win_len, fontsize=16)

