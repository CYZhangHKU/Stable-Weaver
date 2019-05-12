import numpy as np
import scipy.linalg as spl
import scipy.sparse.linalg as spsl

###################################Stable Weaver Algorithm##############################
def WeaverStable(a,b,tDe,tol = 1e-10, maxit = 50000, iteration = False, ini = np.array([-1])):
    '''

    :param a: p_i^a_i, array of {a_i}
    :param b: (delta^T * p)^b_j, array of {b_j}
    :param tDe: Transpose of delta matrix
    :param tol: iteration tolerance
    :param maxit: maximum iteration
    :param iteration: whether to store log-likelihood result.
    :param ini: initial value of p.
    :return: a dict which contains {p_estimation, iteration_results, error}
    '''
    s = sum(a) + sum(b)
    if (ini<=0).any():
        p_sim = np.ones(len(a))
    else:
        p_sim = ini
    p_sim = p_sim / p_sim.sum()
    iterCount = 0
    if (iteration):
        lena = len(a)
        iter = dict()
        iter['path'] = np.zeros((maxit, lena))
        iter['lnLik'] = np.zeros(maxit)
    e = 1e10 #record error

    pos_b_label = np.where(b >= 0)[0]
    neg_b_label = np.where(b < 0)[0]
    b_pos = b[pos_b_label]
    b_neg = b[neg_b_label]
    tDe_pos = tDe[pos_b_label]
    tDe_neg = tDe[neg_b_label]

    while (e>tol) and (iterCount<maxit):
        if(iteration):
            iter['path'][iterCount,:] = p_sim
            iter['lnLik'][iterCount] = sum(a * np.log(p_sim)) + sum(b * np.log(tDe @ p_sim))
        tDe_pos_p = tDe_pos @ p_sim
        tDe_neg_p = tDe_neg @ p_sim

        tau_pos = b_pos / tDe_pos_p
        tau_neg = b_neg / tDe_neg_p

        denom = s - tDe_neg.T @ tau_neg
        num = a + (tDe_pos.T @ tau_pos) * p_sim
        pnew = num / denom
        assert np.all(pnew>=0), 'some p(s) are smaller than 0 during iteration in WeaverBasStable'

        pnew = pnew / sum(pnew)
        e = sum(abs(p_sim - pnew))
        p_sim = pnew
        iterCount += 1

    if (iterCount >= maxit):
       print ("Maxit reached")
    if (iteration):
        iter["path"] = iter['path'][:iterCount,:]
        iter["lnLik"] = iter['lnLik'][:iterCount]
        iter['count'] = {'Weaver':iterCount}
    else:
        iter = dict()
        iter['count'] = {'Weaver':iterCount}
        iter['lnLik'] = sum(a * np.log(p_sim)) + sum(b * np.log(tDe @ p_sim))
    return {'p_est':p_sim, 'iter': iter, 'e':e}

##############################END of Stable Weaver Algorithm############################
##################PYTHON Implementation of Weaver Algorithm#############################

def WeaverBas(a,b,tDe,tol = 1e-10, maxit = 5000, iteration = False, ini = np.array([-1])):
    '''
    :param a: p_i^a_i, array of {a_i}
    :param b: (delta^T * p)^b_j, array of {b_j}
    :param tDe: Transpose of delta matrix
    :param tol: iteration tolerance
    :param maxit: maximum iteration
    :param iteration: whether to store log-likelihood result.
    :param ini: initial value of p.
    :return: a dict which contains {p_estimation, iteration_results, error}
    '''
    s = sum(a) + sum(b)
    if (ini<=0).any():
        p_sim = np.ones(len(a))
    else:
        p_sim = ini

    p_sim = p_sim / sum(p_sim)
    iterCount = 0
    if (iteration):
        lena = len(a)
        iter = dict()
        iter['path'] = np.zeros((maxit, lena))
        iter['lnLik'] = np.zeros(maxit)

    e = 1e10 #record error
    while (e>tol) and (iterCount<maxit):
        if(iteration):
            iter['path'][iterCount,:] = p_sim
            iter['lnLik'][iterCount] = sum(a * np.log(p_sim)) + sum(b * np.log(tDe @ p_sim))
        tau = b / (tDe @ p_sim)
        pnew = a / (s  - tDe.T @ tau)
        pnew += 1e-5 * tol
        assert all(pnew>=0), 'some p(s) are smaller than 0 during iteration in WeaverBas'

        pnew = pnew / sum(pnew)
        e = sum(abs(p_sim - pnew))
        p_sim = pnew.copy()
        iterCount += 1
    if (iterCount >= maxit):
       print ("Maxit reached")
    if (iteration):
        iter["path"] = iter['path'][:iterCount,:]
        iter["lnLik"] = iter['lnLik'][:iterCount]
        iter['count'] = {'Weaver':iterCount}
    else:
        iter = dict()
        iter['count'] = {'Weaver':iterCount}
        iter['lnLik'] = sum(a * np.log(p_sim)) + sum(b * np.log(tDe @ p_sim))
    return {'p_est':p_sim, 'iter': iter, 'e':e}

def WeaverBayes(a, b, tDe, PriorThickness = 0, tol = 1e-10, maxit = 10000, iteration = False, ini = np.array([-1])):
    '''
    :param a: p_i^a_i, array of {a_i}
    :param b: (delta^T * p)^b_j, array of {b_j}
    :param tDe: Transpose of delta matrix
    :param PriorThickness: The setting or prior distribution gamma
    :param tol: iteration tolerance
    :param maxit: maximum iteration
    :param iteration: whether to store log-likelihood result.
    :param ini: initial values of p
    :return: a dict which contains {p_estimation, iteration_results, error}
    '''
    if (a<=0).any():
        maxit = max(50000, maxit)
    if (ini<=0).any():
        p_sim = np.ones(len(a))
    else:
        p_sim = ini

    p_sim = p_sim / sum(p_sim)
    if (p_sim <= 0).any():
        p_sim = np.ones(len(a))/len(a)
    e = 1e10
    if PriorThickness == 0:
        PriorThickness = (sum(abs(b))) * (sum(abs(b))) / ((sum(abs(a))) + 1) * 10
    iterCount = 0
    iterCountWeaverBayes = 0

    if iteration:
        iter = dict()
    while (e > tol) and (iterCount < maxit):
        pnew = WeaverBas(a + p_sim * PriorThickness, b, tDe, tol = tol, iteration = iteration, ini = p_sim)
        if(iteration):
            lnLikOffset = -sum(p_sim * PriorThickness * np.log(p_sim))
            if len(iter) == 0:
                iter = {'path': pnew['iter']['path'], 'lnLik': pnew['iter']['lnLik'] + lnLikOffset}
            else:
                iter['path'] = np.vstack((iter['path'], pnew['iter']['path']))
                real_llk = np.array([])
                for bas_count in range(len(pnew['iter']['path'])):
                    real_llk = np.hstack((real_llk, sum(a * np.log(pnew['iter']['path'][bas_count,:])) + sum(b * np.log(tDe @ pnew['iter']['path'][bas_count,:])) ))
                iter['lnLik'] = np.hstack((iter['lnLik'], real_llk))
        iterCount = iterCount + pnew['iter']['count']['Weaver']
        iterCountWeaverBayes += 1
        pnew = pnew['p_est']
        assert all(pnew >= 0),  'some p(s) are smaller than 0 during iteration in WeaverBayes'
        e = sum(abs(p_sim - pnew))
        p_sim = pnew.copy()
        p_sim = p_sim / sum(p_sim)
    if iterCount >= maxit:
        print ("maxit reached")

    assert all(pnew>=0), 'some p(s) are smaller than 0 during iteration after WeaverBayes'

    if iteration:
        iter['count'] = {"Weaver": iterCount, "WeaverBayes":iterCountWeaverBayes}
    else:
        iter = {'count': {"Weaver": iterCount, "WeaverBayes":iterCountWeaverBayes},
                'lnLik' : sum(a * np.log(p_sim)) + sum(b * np.log(tDe @ p_sim))}
    return {'p_est': p_sim, 'iter':iter, 'prithi': PriorThickness}

def Weaver(a, b, tDe, tol = 1e-10, maxit = 100000, iteration = False, ini = np.array([-1]), PriorThickness = 0):
    res = None
    try:
        res = WeaverBas(a, b, tDe, iteration= iteration, tol = tol, maxit = maxit, ini = ini)
    except Exception as e:
        res = WeaverBayes(a, b, tDe, iteration= iteration, tol = tol, maxit = maxit, ini = ini, PriorThickness = PriorThickness)
    if (res == None) or (np.isnan(res['iter']['lnLik']).sum()>0):
        res = WeaverBayes(a, b, tDe, iteration= iteration, tol = tol, maxit = maxit, ini = ini, PriorThickness = PriorThickness)
    return res

##################END of Weaver Algorithm############################################

########################Iterative Luce Spectral Ranking##############################
def statdist(generator, method="kernel"):   ###The function statdist is from ILSR open source codes.
    ## https://papers.nips.cc/paper/5681-fast-and-accurate-inference-of-plackettluce-models
    """Compute the stationary distribution of a Markov chain.

    The Markov chain is descibed by its infinitesimal generator matrix. It
    needs to be irreducible, but does not need to be aperiodic. Computing the
    stationary distribution can be done with one of the following methods:

    - `kernel`: directly computes the left null space (co-kernel) the generator
      matrix using its LU-decomposition.
    - `eigenval`: finds the leading left eigenvector of an equivalent
      discrete-time MC using `scipy.sparse.linalg.eigs`.
    - `power`: finds the leading left eigenvector of an equivalent
      discrete-time MC using power iterations.
    """
    n = generator.shape[0]
    if method == "kernel":
        # `lu` contains U on the upper triangle, including the diagonal.
        # TODO: this raises a warning when generator is singular (which it, by
        # construction, is! I could add:
        #
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings('ignore')
        #
        # But i don't know what the performance implications are.
        lu, piv = spl.lu_factor(generator.T, check_finite=False)
        # The last row contains 0's only.
        left = lu[:-1,:-1]
        right = -lu[:-1,-1]
        # Solves system `left * x = right`. Assumes that `left` is
        # upper-triangular (ignores lower triangle.)
        res = spl.solve_triangular(left, right, check_finite=False)
        res = np.append(res, 1.0)
        return (1.0 / res.sum()) * res
    if method == "eigenval":
        # DTMC is a row-stochastic matrix.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps*generator
        # Find the leading left eigenvector.
        vals, vecs = spsl.eigs(mat.T, k=1)
        res = np.real(vecs[:,0])
        return (1.0 / res.sum()) * res
    else:
        raise RuntimeError("not (yet?) implemented")

def ILSR(C_ind, A_ind, tDe, weights,  iteration = False, maxit = 5000, tol = 1e-10):
    assert len(C_ind) == len(A_ind)
    types = tDe.shape[0]
    K = tDe.shape[1]
    if weights is None:
        weights = np.ones(len(C_ind))
    p_sim = np.ones(K) / K
    p_store = []
    iterCount = 0
    e = 1e10
    C_ind_mat = tDe[C_ind]
    A_ind_mat = tDe[A_ind]
    A_exc_mat = np.einsum('ij,ik->ijk', A_ind_mat - C_ind_mat, C_ind_mat)
    while (iterCount < maxit) & (e > tol):
        tDe_p = tDe @ p_sim
        tDe_p_A = tDe_p[A_ind]
        if iteration:
            p_store.append(p_sim)
        chain = (A_exc_mat.reshape(len(A_ind), (K * K)).T @ (1.0 / tDe_p_A * weights)).reshape(K, K)
        chain_eig = chain - np.diag(chain.sum(axis=1))
        p_new = statdist(chain_eig)
        e = np.sum(abs(p_new - p_sim))
        p_sim = p_new
        iterCount += 1
    return_list = {'path': p_store, 'e': e, 'iter': iterCount, 'p_est': p_sim}
    return return_list
#################################END of ILSR#########################################
#############################Self-consistency####################################
def self_consistency(C_ind, A_ind, tDe, weights, maxit = 100000, tol = 1e-10, iteration = False):
    assert len(C_ind) == len(A_ind)
    types = tDe.shape[0]
    K = tDe.shape[1]
    p_sim = np.ones(K) / K
    p_store = []
    llk_store = []
    iterCount = 0
    e = 1e10
    C_ind_mat = tDe[C_ind]
    A_ind_mat = tDe[A_ind]


    while (iterCount < maxit) & (e > tol):
        tDe_p = tDe @ p_sim
        tDe_p_A = tDe_p[A_ind]#sum_k (beta_ik*p_k)
        tDe_p_C = tDe_p[C_ind]#sum_k (alpha_ik*p_k)
        if iteration:
            llk_store.append(sum(weights * np.log(tDe_p_C / tDe_p_A)))

        U = (C_ind_mat).T @ (weights / tDe_p_C) * p_sim
        V = (1-A_ind_mat).T @ (weights / tDe_p_A) * p_sim

        M = (U+V).sum()
        p_new = (U+V)/M

        e = np.sum(abs(p_sim - p_new))
        p_sim = p_new.copy()
        iterCount += 1


    return_list = {'path': p_store, 'llk': llk_store, 'e': e, 'iter': iterCount, 'p_est': p_sim}
    return return_list

############################End of self-consistency##############################

#########Likelihood function, score function, hessian matrix of Incomplete multinomial Model####
def likelihood(p,a,b,De):
    if np.any(p<=0):
        return np.inf
    # p = p / sum(p)
    s = sum(a)+sum(b)
    return -(np.sum(a * np.log(p)) + np.sum(b * np.log(De.T @ p))- s * (np.sum(p)-1))
def score(p, a, b, De):
    # p = p / sum(p)
    d = len(a)
    q = len(b)
    s = sum(a) + sum(b)
    f_v = a / p + De @ (b / (De.T @ p)) - s

    return -f_v
def hess(p, a, b, De):
    d = len(a)
    q = len(b)
    tDe_p = De.T @ p
    on_term = a / (p ** 2)
    De_kt = De[:, None, :] * De[None, :, :]
    tau2 = b / (tDe_p ** 2)
    off_term = (De_kt.reshape(d*d, q) @ tau2).reshape(d,d)
    hess_mat=  on_term + off_term
    return hess_mat

########################################END#####################################################