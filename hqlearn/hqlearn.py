from numpy import *
import numpy as np
from collections import OrderedDict

import pandas as pd
import pymc as pm

import kabuki
from kabuki.utils import stochastic_from_dist
from kabuki.hierarchical import Knode

rand = random.random


####Model simulation
def RL_generate(lrate, invtemp, size=100):
    '''Generate data from the model. Takes learning rate, inverse temperature and
    number of triasl. Returns a data array with each line a trial, column 0 stim,
    column 1 choice, column 2 reward'''
    thresholds = [.8, .6]
    V = .5*ones((2,2))
    data = ones((size,3))

    for t in range(size):
        # simple hack for stimuli
        s = t%2
        # choose according to softmax
        a = softmax_choice(invtemp*V[s])

        # implement probabilistic reward
        r = rand()
        if a == s:
            if r < thresholds[s]:
                reward = 1
            else:
                reward = 0

        else:
            if r > thresholds[s]:
                reward = 1
            else:
                reward = 0

        # store data
        data[t,0] = s
        data[t,1] = a
        data[t,2] = reward

        # Actual RL eqution
        V[s,a] = V[s,a] + lrate * (reward - V[s,a])

    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])

    return df


####Model likelihood
def RL_likelihood(data, lrate, invtemp):
    '''Given data (first column is stim, second is action, third is reward)
    lrate and inverse temperature, produces the likelihood of the data given model and params'''

    # initialize problem
    V = .5*ones((2,2))

    logp = 0

    for t in data.index:
        s = data['state'][t]
        a = data['action'][t]
        reward = data['reward'][t]

        # get proba and add it to the log likelihood
        proba = softmax_proba(invtemp*V[s], a)
        logp += np.log(proba)
        V[s,a] = V[s,a] + lrate * (reward - V[s,a])

    return logp

RL_like = stochastic_from_dist(name="QLearn likelihood",
                               logp=RL_likelihood,
                               random=RL_generate)

class HRL(kabuki.Hierarchical):
    def create_knodes(self):
      # value contains the starting value for the group parameter
      invtemp = self.create_family_trunc_normal('invtemp', value=5, lower=0)
      lrate = self.create_family_beta('lrate', value=0.5, g_certainty=2)

      # likelihood
      like = kabuki.Knode(RL_like, 'RL_like', lrate=lrate['lrate_bottom'],
                          invtemp=invtemp['invtemp_bottom'],
                          col_name=['state', 'action', 'reward'],
                          observed=True)

      # return all knodes as a list
      return invtemp.values() + lrate.values() + [like]

    def create_family_trunc_normal(self, name, value=0, lower=None,
                                   upper=None, g_mu=0, g_sigma=10, std_std=2,
                                   var_value=.1):
        """Similar to create_family_normal() but creates a Uniform
        group distribution and a truncated subject distribution.

        See create_family_normal() help for more information.

        """
        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.TruncatedNormal, '%s' % name, a=lower,
                      b=upper, mu=g_mu, tau=g_sigma**-2, value=value, depends=self.depends[name])
            var = Knode(pm.HalfNormal, '%s_var' % name, tau=std_std**-2, value=var_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g,
                         tau=tau, a=lower, b=upper, value=value,
                         depends=('subj_idx',), subj=True, plot=self.plot_subjs)

            knodes['%s'%name] = g
            knodes['%s_var'%name] = var
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Uniform, name, lower=lower,
                         upper=upper, value=value,
                         depends=self.depends[name])
            knodes['%s_bottom'%name] = subj

        return knodes

    def create_family_beta(self, name, value=.5, g_value=.5, g_mean=.5, g_certainty=2,
                           var_alpha=1, var_beta=1, var_value=.1):
        """Similar to create_family_normal() but beta for the subject
        and group mean nodes. This is useful when the parameter space
        is restricted from [0, 1].

        See create_family_normal() help for more information.

        """

        knodes = OrderedDict()
        if self.is_group_model and name not in self.group_only_nodes:
            g_mean = Knode(pm.Beta, name, alpha=g_mean*g_certainty, beta=(1-g_mean)*g_certainty,
                      value=g_value, depends=self.depends[name])

            g_certainty = Knode(pm.Gamma, '%s_certainty' % name,
                                alpha=var_alpha, beta=var_beta, value=var_value)

            alpha = Knode(pm.Deterministic, '%s_alpha' % name, eval=lambda mean, certainty: mean*certainty,
                      mean=g_mean, certainty=g_certainty, plot=False, trace=False, hidden=True)

            beta = Knode(pm.Deterministic, '%s_beta' % name, eval=lambda mean, certainty: (1-mean)*certainty,
                      mean=g_mean, certainty=g_certainty, plot=False, trace=False, hidden=True)

            subj = Knode(pm.Beta, '%s_subj'%name, alpha=alpha, beta=beta,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=False)

            knodes['%s'%name]            = g_mean
            knodes['%s_certainty'%name]  = g_certainty
            knodes['%s_alpha'%name]      = alpha
            knodes['%s_beta'%name]       = beta
            knodes['%s_bottom'%name]     = subj

        else:
            g = Knode(pm.Beta, name, alpha=g_mean*g_certainty, beta=(1-g_mean)*g_certainty, value=value,
                      depends=self.depends[name])

            knodes['%s_bottom'%name] = g

        return knodes


def check_params_valid(**params):
    lrate = params.get('lrate', .1)
    invtemp = params.get('invtemp', 10)

    return (0 < lrate) and (lrate < 1) and (invtemp > 0)

def gen_data(params=None, **kwargs):
    if params is None:
        params = {'lrate': .1, 'invtemp': 10}

    return kabuki.generate.gen_rand_data(RL_generate, params,
						      check_valid_func=check_params_valid,
						      **kwargs)

def softmax_choice(V1):
    '''Taking values to compare, returns choice and proba of choice according
    to softmax on those values'''

    p1 = math.exp(V1[0])
    p1 = p1/(p1+math.exp(V1[1]))

    r2 = rand()
    if r2 < p1:
        choice = 0
        proba = p1
    else:
        choice = 1
        proba = 1 - p1

    return choice

def softmax_proba(V1, a):
    '''Return softmax proba of the chosen action'''

    p1 = math.exp(V1[a])
    p1 = p1/(p1+math.exp(V1[1-a]))

    return p1
