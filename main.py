from Encoder import *
import random
import logging
import sys

import torch
import torch.nn as nn
import numpy as np

def test():
    encoder = EncoderRNN(invoc_size=10, vector_size=13, hidden_size=20)
    print(encoder)
    hidden = encoder.init_hidden()
    input_words = torch.LongTensor([1, 2, 3, 4])
    output, hidden = encoder(V(input_words), hidden)
    print('Output size:', output.size())
    print('Hidden size:', [h.size() for h in hidden])
    print(str(hidden))
    print(str(output))

if __name__ == '__main__':
    test()
