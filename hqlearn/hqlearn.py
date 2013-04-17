import numpy as np
from collections import OrderedDict

import pandas as pd
import pymc as pm

import kabuki
from kabuki.utils import stochastic_from_dist
from kabuki.hierarchical import Knode
import kabuki.step_methods as steps

rand = np.random.random


####Model simulation
def RL_generate(lrate, invtemp, num_states=2, reward_probs=None, size=500):
    '''Generate data from the model. Takes learning rate, inverse temperature and
    number of trials. Returns a data array with each line a trial, column 0 stim,
    column 1 choice, column 2 reward'''
    if reward_probs is None:
        reward_probs = np.array([[.8, .6, .4],
                                 [.4, .2, .0]])
    states, actions = reward_probs.shape
    V = 1./actions * np.ones_like(reward_probs)

    data = np.ones((size, 3))

    for t in range(size):
        # simple hack for stimuli
        s = t % states
        # choose according to softmax
        a = softmax_choice(V[s,:], invtemp)

        # implement probabilistic reward
        reward = rand() < reward_probs[s, a]

        # store data
        data[t, 0] = s
        data[t, 1] = a
        data[t, 2] = reward

        # Actual RL eqution
        V[s,a] += lrate * (reward - V[s,a])


    df = pd.DataFrame(data, columns=['state', 'action', 'reward'])

    return df


####Model likelihood
def RL_likelihood(data, lrate, invtemp):
    '''Given data (first column is stim, second is action, third is reward)
    lrate and inverse temperature, produces the likelihood of the data given model and params'''

    # initialize problem
    states = len(data.state.unique())
    actions = len(data.action.unique())
    V = 1./actions * np.ones((states, actions))

    logp = 0
    for t, (s, a, reward) in data[['state', 'action', 'reward']].iterrows():
        # get proba and add it to the log likelihood
        proba = softmax_proba(V[s,:], a, invtemp)
        logp += np.log(proba)
        V[s, a] += lrate * (reward - V[s, a])

    return logp

RL_like = stochastic_from_dist(name="QLearn likelihood",
                               logp=RL_likelihood,
                               random=RL_generate)

def check_params_valid(**params):
    lrate = params.get('lrate', .1)
    invtemp = params.get('invtemp', 5)

    return (0 < lrate) and (lrate < 1) and (invtemp > 0)

def gen_data(params=None, **kwargs):
    if params is None:
        params = {'lrate': .1, 'invtemp': 10}

    return kabuki.generate.gen_rand_data(RL_generate, params,
                                         check_valid_func=check_params_valid,
                                         **kwargs)

def softmax_choice(V, invtemp):
    '''Taking values to compare, returns choice and proba of choice according
    to softmax on those values'''

    p_choices = np.empty_like(V)
    for a in range(len(V)):
        p_choices[a] = softmax_proba(V, a, invtemp)

    choice = np.random.multinomial(1, p_choices).nonzero()[0][0]

    return choice

def softmax_proba(V, a, invtemp):
    '''Return softmax proba of the chosen action'''
    return np.exp(invtemp*V[a]) / np.sum(np.exp(invtemp*V))

class HRL(kabuki.Hierarchical):
    def __init__(self, *args, **kwargs):
        super(HRL, self).__init__(*args, **kwargs)

        self.slice_widths = {'invtemp':1, 'lrate':0.05, 'invtemp_std': 1, 'lrate_std': 0.15}

    def create_knodes(self):
        # value contains the starting value for the group parameter
        invtemp = self._create_family_gamma_gamma_hnormal('invtemp', value=5, g_mean=5)
        lrate = self._create_family_beta('lrate', value=0.5, g_certainty=2)

         # likelihood
        like = kabuki.Knode(RL_like, 'RL_like', lrate=lrate['lrate_bottom'],
                            invtemp=invtemp['invtemp_bottom'],
                            col_name=['state', 'action', 'reward'],
                            observed=True)

        # return all knodes as a list
        return invtemp.values() + lrate.values() + [like]

    def _create_family_trunc_normal(self, name, value=0, lower=None,
                                   upper=None, std_lower=1e-10,
                                   std_upper=100, std_value=.1):
        """Similar to _create_family_normal() but creates a Uniform
        group distribution and a truncated subject distribution.

        See _create_family_normal() help for more information.

        """
        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Uniform, '%s' % name, lower=lower,
                      upper=upper, value=value, depends=self.depends[name])
            std = Knode(pm.Uniform, '%s_std' % name, lower=std_lower,
                        upper=std_upper, value=std_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=std,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g,
                         tau=tau, a=lower, b=upper, value=value,
                         depends=('subj_idx',), subj=True, plot=self.plot_subjs)

            knodes['%s'%name] = g
            knodes['%s_std'%name] = std
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Uniform, name, lower=lower,
                         upper=upper, value=value,
                         depends=self.depends[name])
            knodes['%s_bottom'%name] = subj

        return knodes

    def _create_family_gamma_gamma_hnormal(self, name, value=1, g_mean=1, g_std=1, std_std=2, std_value=.1):
        """Similar to _create_family_normal_normal_hnormal() but adds an exponential
        transform knode to the subject and group mean nodes. This is useful
        when the parameter space is restricted from [0, +oo).

        See _create_family_normal_normal_hnormal() help for more information.

        """

        knodes = OrderedDict()
        g_shape = (g_mean**2) / (g_std**2)
        g_rate = g_mean / (g_std**2)
        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Gamma, name, alpha=g_shape, beta=g_rate,
                            value=g_mean, depends=self.depends[name])

            std = Knode(pm.HalfNormal, '%s_std' % name, tau=std_std**-2, value=std_value)

            shape = Knode(pm.Deterministic, '%s_shape' % name, eval=lambda x,y: (x**2)/(y**2),
                        x=g, y=std, plot=False, trace=False, hidden=True)

            rate = Knode(pm.Deterministic, '%s_rate' % name, eval=lambda x,y: x/(y**2),
                        x=g, y=std, plot=False, trace=False, hidden=True)


            subj = Knode(pm.Gamma, '%s_subj'%name, alpha=shape, beta=rate,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=False)

            knodes['%s'%name]            = g
            knodes['%s_std'%name]        = std
            knodes['%s_rate'%name]       = rate
            knodes['%s_shape'%name]      = shape
            knodes['%s_bottom'%name]     = subj

        else:
            g = Knode(pm.Gamma, name, alpha=g_shape, beta=g_rate, value=value,
                            depends=self.depends[name])

            knodes['%s_bottom'%name] = g

        return knodes

    def _create_family_beta(self, name, value=.5, g_value=.5, g_mean=.5, g_certainty=2,
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


    def pre_sample(self, use_slice=True):
        for name, node_descr in self.iter_stochastics():
            node = node_descr['node']
            if isinstance(node, pm.Normal) and np.all([isinstance(x, pm.Normal) for x in node.extended_children]):
                self.mc.use_step_method(steps.kNormalNormal, node)
            else:
                knode_name = node_descr['knode_name'].replace('_subj', '')
                if knode_name in ['st', 'sv', 'sz']:
                    left = 0
                else:
                    left = None
                self.mc.use_step_method(steps.SliceStep, node, width=self.slice_widths.get(knode_name, 1),
                                        left=left, maxiter=5000)
