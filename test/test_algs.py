import numpy as np
from nets import filter_negs
from nets import neural_net
import train_nets as tn

def test_filtering():
    assert filter_negs.remove_strings_w_subs(['hello','lets','find','some','substrings'], ['ll','so']) == ['lets','find','substrings']

def test_listcomp():
    testvec = [[0],[1],[0],[1],[0],[1]]
    assert [ i for i, response in enumerate(testvec) if response == [1] ] == [1,3,5]

def test_dna():
    assert tn.get_kmers('ATCG',2) == ['AT','TC','CG']
    assert tn.encode('ACTGCT') == [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0], [0,0,1,0]]
    assert filter_negs.reverse_complement("TCGGGCCC") == "GGGCCCGA"
    assert tn.decode([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0], [0,0,1,0]]) == ['ACTGCT']
