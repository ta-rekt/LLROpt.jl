import numpy as np
import itertools
import ctmc
import utils
import plot
import fisher

def test_mfpt_part():

    #                 kboff, konu,  kbon,  koffu, kuoff, konb,  kuon,  koffb
    edges = np.array([(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]) - 1
    input_edges = [(0,2), (1,3)]
    on_ind = [0,1,0,1]
    c = 1
    weights = np.ones(8)
    weighted_edges = np.array([(u,v,w) for ((u,v),w) in zip(edges, weights)])

    tauon, tauoff = ctmc.mean_fpt_part(weighted_edges, input_edges, c, on_ind)

    print('tauon:', tauon, 'tauoff:', tauoff)
    assert tauon == 3
    assert tauoff == 2

    # 10010001111110101001000101010101000000011100000111111111010001100010100

test_mfpt_part()