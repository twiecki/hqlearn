import unittest

import hqlearn

class TestModel(unittest.TestCase):
    def setUp(self):
        self.true_params = {'lrate': .1, 'invtemp': 10}

    def test_model_no_group_breakdown(self):
        data, _ = hqlearn.gen_data(self.true_params, subjs=1)
        m = hqlearn.HRL(data)
        m.sample(50, 10)

    def test_model_group_breakdown(self):
        data, _ = hqlearn.gen_data(self.true_params, subjs=12)
        m = hqlearn.HRL(data)
        m.sample(50, 10)