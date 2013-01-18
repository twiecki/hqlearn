from numpy import *
import numpy as np

import pymc as pm
import kabuki
import pandas as pd

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
    # get number of trials
    size = len(data)

    likelihood = 0

    for t in range(size):
        s = data['state'][t]
        a = data['action'][t]
        # get proba and add it to the log likelihood
        proba = softmax_proba(invtemp*V[s], a)
        likelihood = likelihood + math.log(proba)

        reward = data['reward'][t]
        V[s,a] = V[s,a] + lrate * (reward - V[s,a])

    # this is actually a log likelihood
    return likelihood

RL_like = pm.stochastic_from_dist(name="QLearn likelihood",
                                  logp=RL_likelihood,
                                  random=RL_generate,
                                  dtype=np.dtype('O'), #pymc hack
                                  mv=False)


class HRL(kabuki.Hierarchical):
    def create_knodes(self):
      # Create family of 4 knodes, mu, sigma, subj and transform
      # returns a dictionary mapping the name, appended with a key
      # like _var, _subj or _bottom to the corresponding Knode object.
      # Of specific interest is the _bottom knode as that's what we'll
      # put into the likelihood
      invtemp = self.create_family_exp('invtemp', value=1)
      # value contains the starting value for the group parameter
      lrate = self.create_family_invlogit('lrate', value=0.5) # will automatically convert .5 to logit

      # likelihood
      like = kabuki.Knode(RL_like, 'RL_like', lrate=lrate['lrate_bottom'],
				       invtemp=invtemp['invtemp_bottom'],
				       col_name=['state', 'action', 'reward'],
				       observed=True)
      # return all knodes as a list
      return invtemp.values() + lrate.values() + [like]

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
