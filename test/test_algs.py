import numpy as np
from nets import filter_negs

def test_filtering():
    assert remove_strings_w_subs(['hello','lets','find','some','substrings'], ['ll','so']) == ['lets','find','substrings']
