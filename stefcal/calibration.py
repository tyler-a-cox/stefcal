import numpy as np


import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)
from hera_cal import abscal, redcal, datacontainer


@jax.jit
def stefcal_optimizer(data_matrix, model_matrix, weights, tol=1e-10, maxiter=1000, stepsize=0.5):
    """
    Function to run stefcal optimization

    Parameters:
    ----------
    data_matrix: np.ndarray
        Data matrix (nants, nants)
    model_matrix: np.ndarray
        Model matrix (nants, nants)
    weights: np.ndarray
        Weights matrix (nants, nants)
    tol: float, optional, default=1e-10
        Tolerance for convergence criterea
    maxiter: int, optional, default=1000
        Maximum number of iterations to run
    stepsize: float, optional, default=0.5
        Step size for the optimization loop

    Returns:
    -------
    gains: np.ndarray
        Gains
    niters: int
        Number of iterations
    conv_crit: float
        Convergence criterea
    """
    def inner_function(args):
        """
        Main optimization loop
        """
        gains, i, tau = args
        g_old = jnp.copy(gains)
        zg = gains[:, None] * model_matrix
        zgw = zg * weights

        # Compute gains
        gains = jnp.sum(jnp.conj(data_matrix) * zgw, axis=(0)) / jnp.sum(jnp.conj(zgw) * zg, axis=(0))

        # Set gains to 1 if they are nan
        gains = jnp.where(jnp.isnan(gains), 1, gains)
        gains = gains * stepsize + g_old * (1 - stepsize)
        
        # Compute convergence criterea
        tau = jnp.sqrt(jnp.sum(jnp.abs(gains - g_old) ** 2))/ jnp.sqrt(jnp.sum(jnp.abs(gains)**2))
        return gains, i + 1, tau
    
    def conditional_function(args):
        """
        Conditional function to check to convergence criterea
        """
        _, i, tau = args
        return (tau > tol) & (i < maxiter)
    
    nants = data_matrix.shape[0]
    gains = jnp.ones(nants, dtype=complex)
    
    return jax.lax.while_loop(conditional_function, inner_function, (gains, 0, 1))

def build_model_matrices(data, model, flags, baselines, antenna_flags={}):
    """
    Function to build model matrices for stefcal

    Parameters:
    ----------
    data: dict
        Dictionary of data
    model: dict
        Dictionary of model
    flags: dict
        Dictionary of flags
    data_bls: list
        List of data baselines
    model_bls: list
        List of model baselines
    
    Returns:
    -------
    data_matrix: np.ndarray
        Data matrix
    model_matrix: np.ndarray
        Model matrix
    wgts_matrix: np.ndarray
        Weights matrix
    map_ants_to_index: dict
        Dictionary mapping antennas to indices within the data, model, and wgts matrices
    """
    # Get unique antennas
    ants = sorted(list(set(sum([list(k[:2]) for k in data], []))))
    pols = sorted(list(set([k[2] for k in data_bls])))

    # Remove flagged antennas
    ants = [ant for ant in ants if not antenna_flags.get((ant, 'J' + pols[0]), False)]
    nants = len(ants)

    # Map antennas to indices in the data, model, and wgts matrices
    map_ants_to_index = {
        ant: ki for ki, ant in enumerate(ants)
    }
    
    # Number of times and frequencies
    ntimes, nfreqs = data[baselines[0]].shape
    
    # Populate matrices
    data_matrix = np.zeros((nants, nants, ntimes, nfreqs), dtype='complex')
    wgts_matrix = np.zeros((nants, nants, ntimes, nfreqs), dtype='float')
    model_matrix = np.zeros((nants, nants, ntimes, nfreqs), dtype='complex')
    
    for bl in baselines:
        m, n = map_ants_to_index[bl[0]], map_ants_to_index[bl[1]]
        
        # Data matrix
        data_matrix[m, n] = data[bl]
        data_matrix[n, m] = data[bl].conj()
        
        # Weights matrix
        wgts_matrix[m, n] = (~flags[bl]).astype(float)
        wgts_matrix[n, m] = wgts_matrix[m, n]
        
        # Model Matrix
        model_matrix[m, n] = model[bl]
        model_matrix[n, m] = model[bl].conj()

    return data_matrix, model_matrix, wgts_matrix, map_ants_to_index

def sky_calibration(data, model, flags, ant_flags={}, tol=1e-10, maxiter=1000, stepsize=0.5):
    """
    Function to run stefcal for a dictionary of data and model

    Parameters:
    ----------
    data: dict
        Dictionary of data
    model: dict
        Dictionary of model
    flags: dict
        Dictionary of flags
    ant_flags: dict, optional, default={}
        Dictionary of antenna flags
    tol: float, optional, default=1e-10
        Tolerance for convergence
    maxiter: int, optional, default=1000
        Maximum number of iterations
    stepsize: float, optional, default=0.5
        Step size for the optimization

    Returns:
    -------
    gain_dict: dict
        Dictionary of gains
    niters: np.ndarray
        Number of iterations
    conv_crits: np.ndarray
        Convergence criterea
    """
    # Get number of times and frequencies
    ntimes, nfreqs = data[list(data.keys())[0]].shape

    # Get unique polarizations in the data
    pols = sorted(list(set([k[2] for k in data.keys()])))

    # Get antennas
    ants = sorted(list(set(sum([list(k[:2]) for k in data.keys()], []))))

    # get keys from model and data dictionary
    if isinstance(model, datacontainer.RedDataContainer):
        all_bls = sorted(set(data.keys()))
    else:
        all_bls = sorted(set(data.keys()) & set(model.keys()))
        
    # Store gains and metadata
    gains = {}
    niters = {}
    conv_crits = {}
    
    for pol in pols:
        # Initialize arrays for gains, niters, and convergence criterea
        gain_array = []
        niter_array = []
        conv_crit_array = []

        # Get data baselines
        baselines = [k for k in all_bls if k[2] == pol]

        # Pack data and model into numpy arrays
        data_matrix, model_matrix, wgts, map_ants_to_index = build_model_matrices(
            data, model, flags, baselines, antenna_flags=ant_flags
        )

        for ti in range(ntimes):
            _gains = []
            _niters = []
            _conv_crits= []
            for fi in range(nfreqs):
                if wgts[..., ti, fi].sum() > 0:
                    gain, niter, conv_crit = stefcal_optimizer(
                        data_matrix[..., ti, fi], model_matrix[..., ti, fi], wgts[..., ti, fi], 
                        tol=tol, maxiter=maxiter, stepsize=stepsize
                    )
                    _niters.append(niter)
                    _conv_crits.append(conv_crit)
                    _gains.append(gain)
                    
                else:
                    gain = np.ones(data_matrix.shape[2:], dtype='complex')
                    _niters.append(0)
                    _conv_crits.append(np.nan)
                    _gains.append(gain)
                    
            gain_array.append(_gains)
            niter_array.append(_niters)
            conv_crit_array.append(_conv_crits)
        
        gain_array = np.array(gain_array)
    
        for k in ants:
            if k in map_ants_to_index:
                gains[(k, "J" + pol)] = gain_array[..., map_ants_to_index[k]]
            else:
                gains[(k, "J" + pol)] = np.ones((ntimes, nfreqs), dtype='complex')

        niters[pol] = np.array(niter_array)
        conv_crits[pol] = np.array(conv_crit_array)
                
    return gains, niters, conv_crits
                