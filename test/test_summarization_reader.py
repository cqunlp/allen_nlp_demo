import unittest
import os
import torch
import numpy as np

from summarization.cnndm_reader import CNN_DM_Reader
from allennlp.common.util import START_SYMBOL,END_SYMBOL
class TestReaders(unittest.TestCase):

    def test_cnn_dm_reader(self):
        reader = CNN_DM_Reader(cnn_tokenized_dir='/Users/a408/PythonProjects/cnndm_dataset/cnn/stories',separate_namespaces=False)

        dataset = reader.read('/Users/a408/PythonProjects/cnn-dailymail/url_lists/cnn_wayback_test_urls.txt')
        for sample in dataset:
            self.assertEqual(sample.fields["source_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["source_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["source_tokens"]), 2)

            self.assertEqual(sample.fields["target_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["target_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["target_tokens"]), 2)


