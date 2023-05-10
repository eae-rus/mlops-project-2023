import unittest
import numpy as np
from comtrade import Comtrade
from src.data.signal_dataset import SignalDataset


class SignalDatasetTest(unittest.TestCase):
    def test_normalize_data(self):
        rec = Comtrade()
        signal_dataset = SignalDataset('', 0, rec)
        self.assertEqual(signal_dataset.normalize_data(np.array([100.]), 5), np.array([20.]))

    def test_normalize_data_1(self):
        rec = Comtrade()
        signal_dataset = SignalDataset('', 0, rec)
        test_string_list = ['get', 'channel_index', 'name']
        self.assertEqual(signal_dataset.get_channel_index_by_name(test_string_list, 'channel'), -1)


if __name__ == '__main__':
    unittest.main()
