import numpy as np
from nets import filter_negs

def test_filtering():
    assert filter_negs.remove_strings_w_subs(['hello','lets','find','some','substrings'], ['ll','so']) == ['lets','find','substrings']

def test_revcomp():
    assert filter_negs.reverse_complement("TCGGGCCC") == "GGGCCCGA"
