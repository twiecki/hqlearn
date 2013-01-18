import unittest

import hqlearn

class TestGenerate(unittest.TestCase):
    def test_likelihood_correct_min(self):
        true_params = {'lrate': .1, 'invtemp': 10}
        # Making the Data
        data = hqlearn.RL_generate(size=1000, **true_params)

        # Get a likelihood
        lh_true = hqlearn.RL_likelihood(data, **true_params)

        lh_false1 = hqlearn.RL_likelihood(data, .4, 10)

        lh_false2 = hqlearn.RL_likelihood(data, .1, 1)

        self.assertTrue(lh_true > lh_false1)
        self.assertTrue(lh_true > lh_false2)

    def test_gen_data_breakdown(self):
        hqlearn.gen_data(subjs=10)
        hqlearn.gen_data(subjs=1)
        hqlearn.gen_data(size=100)
