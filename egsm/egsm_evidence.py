import numpy as np
import numpy.linalg as la
import egsm_Helper as h



def compute_alpha(basis, data, beta=None, ortho=False):
    """
    Compute the hyper-parameter alpha
    """
    N = np.count_nonzero(~np.isnan(data)) #Number of valid data points
    M = len(basis[0]) # number of components

    pixel = data[~np.isnan(data)]
    basis_rm = np.delete(basis, np.argwhere(np.isnan(data)), 0)
    if ortho == True:
        basis_eval_ = h.gram_schmidt(basis_rm)
        basis_eval = np.array(basis_eval_).T
    else:
        basis_eval = basis_rm

    if beta == None:
        beta_init = (1/np.std(pixel))**2 # variance of a pixel
    else:
        beta_init = beta

    aloops = 0
    maxaloops = 1000
    alpha_init = 1000 ## some starting point
    alpha = alpha_init * 0.8
    alpha_opt = alpha_init

    while (abs(alpha-alpha_opt) > alpha_opt*0.0001) and (aloops < maxaloops):
        aloops = aloops + 1
        alpha = alpha_opt
        A = alpha*np.identity(M) + beta_init*np.dot(basis_eval.T, basis_eval)
        mN_a = np.dot(beta_init*(np.dot(basis_eval,la.inv(A))).T, pixel)
        lamb_a = la.eigvals(beta_init*np.dot(basis_eval.T, basis_eval))
        gamma_a = np.sum(lamb_a/(alpha+lamb_a))
        alpha_opt = gamma_a / np.dot(mN_a.T, mN_a)

    return alpha_opt

def compute_beta(basis, data, inv_Nnu, alpha=None, ortho=False):
    """
    For non-identity covariance matrix
    """
    N = np.count_nonzero(~np.isnan(data)) #Number of valid data points
    M = len(basis[0]) # number of components

    pixel = data[~np.isnan(data)]
    ratio = ratio_array(data, inv_Nnu)
    ratio_rm = np.diag(ratio[~np.isnan(data)])
    basis_rm = np.delete(basis, np.argwhere(np.isnan(data)), 0)
    if ortho == True:
        basis_eval_ = h.gram_schmidt(basis_rm)
        basis_eval = np.array(basis_eval_).T
    else:
        basis_eval = basis_rm

    if alpha == None:
        alpha = get_numerical_alpha(basis, data, ortho=ortho)
    else:
        pass
    beta_init = (1/np.std(pixel))**2 #good first guess
    bloops =0
    maxbloops = 1000
    beta = beta_init*0.8
    beta_opt = beta_init

    while (abs(beta-beta_opt) > beta_opt*0.0001) and (bloops < maxbloops):
        bloops = bloops + 1
        beta = beta_opt

        B = alpha*np.identity(M) + beta*np.dot(basis_eval.T, ratio_rm).dot(basis_eval)
        try:
            mN_b = np.dot(beta*(np.dot(basis_eval,la.inv(B))).T, pixel)
        except la.linalg.LinAlgError:
            #print ("#comp", x_ortho.shape, "pass")
            beta_opt = np.NaN
        lamb_b = la.eigvals(beta*np.dot(basis_eval.T, ratio_rm).dot(basis_eval))
        gamma_b = np.sum(lamb_b/(alpha+lamb_b))
        beta_opt = 1/(np.sum((pixel - np.dot(mN_b, basis_eval.T))**2) / (N-gamma_b))

    return beta_opt


def compute_evidence(basis, data, inv_Nnu, alpha=None, ortho=False):
    # Calculate the evidence using the N-dim Gaussian integration
    """parameters
    M: order of model, N: # of training data, alpha & beta: precision parameters
    
    """
    N = np.count_nonzero(~np.isnan(data)) #Number of valid data points
    M = len(basis[0])
    nan_index = np.where(np.isnan(data))[0]
    pixel = data[~np.isnan(data)]
    basis_rm = np.delete(basis, np.argwhere(np.isnan(data)),0)
    invNnu_rm = np.diag(np.delete(inv_Nnu, nan_index))
    if ortho == True:
        basis_eval_ = h.gram_schmidt(basis_rm)
        basis_eval = np.array(basis_eval_).T
    else:
        basis_eval = basis_rm
    if alpha == None:
        alpha = get_numerical_alpha(basis, data, ortho=ortho)
    else:
        pass

    sigma = alpha*np.identity(M) + (basis_eval.T).dot(invNnu_rm).dot(basis_eval)
    inv_sigma = la.inv(sigma)

    dNd = np.dot(pixel.T, invNnu_rm).dot(pixel)
    dNc = np.dot(pixel.T, invNnu_rm).dot(basis_eval).reshape(1,M)

    log_result = (1/2)*np.log(la.det(invNnu_rm)) + (M/2)*np.log(alpha) - (N/2)*np.log(2*np.pi) \
    - (1/2)*np.trace(np.log(np.abs(sigma))) +(1/2)*np.dot(dNc, inv_sigma).dot(dNc.T)\
    - (1/2)*dNd

    return np.sum(log_result)


def compute_evidence_betaopt(basis, data, inv_Nnu, alpha=None, beta_const=None, ortho=False):
    # Calculate the evidence using the N-dim Gaussian integration
    """parameters
    M: order of model, N: # of training data, alpha & beta: precision parameters
    
    """
    N = np.count_nonzero(~np.isnan(data)) #Number of valid data points
    M = len(basis[0])
    nan_index = np.where(np.isnan(data))[0]
    pixel = data[~np.isnan(data)]
    basis_rm = np.delete(basis, np.argwhere(np.isnan(data)),0)
    ratio = ratio_array(data, inv_Nnu)
    ratio_rm = np.diag(ratio[~np.isnan(data)])

    if ortho == True:
        basis_eval_ = h.gram_schmidt(basis_rm)
        basis_eval = np.array(basis_eval_).T
    else:
        basis_eval = basis_rm
    if alpha == None:
        alpha = get_numerical_alpha(basis, data, ortho=ortho)
    else:
        pass

    if beta_const == None:
        beta = get_numerical_beta(basis, data, inv_Nnu, alpha, ortho=ortho)
    else:
        beta = beta_const

    sigma = alpha*np.identity(M) + beta*np.dot(basis_eval.T, ratio_rm).dot(basis_eval)
    inv_sigma = la.inv(sigma)
    invNnu_rm = beta * ratio_rm

    dNd = np.dot(pixel.T, invNnu_rm).dot(pixel)
    dNc = np.dot(pixel.T, invNnu_rm).dot(basis_eval).reshape(1,M)

    log_result = (1/2)*np.log(la.det(invNnu_rm)) + (M/2)*np.log(alpha) - (N/2)*np.log(2*np.pi) \
    - (1/2)*np.trace(np.log(np.abs(sigma))) +(1/2)*np.dot(dNc, inv_sigma).dot(dNc.T)\
    - (1/2)*dNd

    return np.sum(log_result)