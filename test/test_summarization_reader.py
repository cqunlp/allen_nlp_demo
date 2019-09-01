import unittest
import os
import torch
import numpy as np

from summarization.cnndm_reader import CNN_DM_Reader
from allennlp.common.util import START_SYMBOL,END_SYMBOL
class TestReaders(unittest.TestCase):

    def test_cnn_dm_reader(self):
        reader = CNN_DM_Reader('/home/lv/dataset/CNN_DM_origin/cnn_stories/cnn/stories','/home/lv/dataset/CNN_DM_origin/dailymail_stories/dailymail/stories')

        dataset = reader.read('/home/lv/PycharmProjects/cnn-dailymail/url_lists/all_val.txt')
        for sample in dataset:
            self.assertEqual(sample.fields["source_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["source_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["source_tokens"]), 2)

            self.assertEqual(sample.fields["target_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["target_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["target_tokens"]), 2)


