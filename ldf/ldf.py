import numpy as np
from typing import List
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def _first_layer(agent_loglikelihoods: np.ndarray, alphas: np.ndarray, is_argmax, c):
    T = agent_loglikelihoods.shape[1]
    n_agents = int(agent_loglikelihoods.shape[0])
    pi_t = np.ones((len(alphas), n_agents)) / n_agents
    prob = np.zeros((T, len(alphas), n_agents))
    loglikehood = np.zeros((len(alphas), T))

    for t in range(T):
        for j, alpha in enumerate(alphas):
            if is_argmax:
                pi_t[j] = alpha * pi_t[j]
                p = np.zeros_like(pi_t[j])
                p[pi_t[j].argmax()] = 1
                prob[t, j] = p.copy()
                loglikehood[j, t] = (agent_loglikelihoods[:, t] * p).sum()
                pi_t[j] = pi_t[j] + agent_loglikelihoods[:, t]
            else:
                pi_t[j] = (pi_t[j] ** alpha + c) / (np.sum(pi_t[j] ** alpha) + c)  # pi_t|t-1
                prob[t, j] = pi_t[j].copy()
                loglikehood[j, t] = np.log((np.exp(agent_loglikelihoods[:, t]) * pi_t[j]).sum())
                pi_t[j] = pi_t[j] * np.exp(agent_loglikelihoods[:, t])
                pi_t[j] = pi_t[j] / np.sum(pi_t[j])  # pi_t|t

    return loglikehood, prob


@nb.jit(nopython=True, fastmath=True)
def _second_layer(loglikehood: np.ndarray, deltas: np.ndarray, is_argmax, c):

    T = loglikehood.shape[1]
    res_weights = np.empty((len(deltas), T, loglikehood.shape[0]))
    res_likelihoods = np.empty((len(deltas), T))

    for i, delta in enumerate(deltas):
        weights = np.zeros((T, loglikehood.shape[0]))
        momentum = np.empty(T)
        cumsum = np.zeros((T, loglikehood.shape[0]))

        for t in range(T):
            if is_argmax:
                p = np.zeros(loglikehood.shape[0])
                p[np.argmax(cumsum[t-1])] = 1
                weights[t] = p
                momentum[t] = np.log((np.exp(loglikehood[:, t]) * p).sum())
            else:
                m = max(cumsum[t-1])
                p = np.exp(cumsum[t-1] - np.log(np.sum(np.exp(cumsum[t-1] - m))) - m)
                weights[t] = p
                momentum[t] = np.log((np.exp(loglikehood[:, t]) * p).sum())
            cumsum[t] = delta * cumsum[t-1] + loglikehood[:, t]

        res_likelihoods[i] = momentum
        res_weights[i] = weights

    return res_likelihoods, res_weights


def ldf(agent_loglikelihoods: np.ndarray, levels, discount_factors: List, activation_functions=None, c=10**-20):
    """

    :param agent_loglikelihoods:
    :param levels:
    :param discount_factors:
    :param activation_functions:
    :param c:
    :return:
    """
    alphas = discount_factors[0]
    assert len(discount_factors) == levels
    if not (isinstance(alphas, list) or isinstance(alphas, np.ndarray)):
        alphas = np.array([alphas])

    is_argmax = activation_functions[0] == "argmax"
    loglikehood ,prob = _first_layer(agent_loglikelihoods, alphas, is_argmax, c)
    prob = prob.transpose(1, 0, 2)

    if levels == 1:
        return {"logscores": loglikehood, "weights": prob, "params_weights": None}
    else:
        for k in range(levels-1):
            deltas = discount_factors[k+1]
            if not (isinstance(deltas, list) or isinstance(deltas, np.ndarray)):
                deltas = np.array([deltas])

            is_argmax = activation_functions[k+1] == "argmax"
            loglikehood, param_weights = _second_layer(loglikehood, deltas, is_argmax, c)
            weights = []

            for k in range(len(deltas)):  # iteration over discount factors
                weights.append(
                    np.asarray([(prob[:, i, :].T * j).T.sum(axis=0) for i, j in enumerate(param_weights[k])])) # iteration over T
            prob = np.asarray(weights)
        return {"logscores": loglikehood, "weights": prob, "params_weights": param_weights}