"""
Some sanity checking of polarisation/phasing options for different processors
"""

import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflow", "scripts"))
import ts_simulators
import ts_processors

def test_genotypes_and_distances():
    one_pop = ts_simulators.VariablePopulationSize({})
    ts, theta = one_pop(10)
    
    # phased
    unpol = ts_processors.genotypes_and_distances({"polarised": False, "phased": True})
    unpol_feat = unpol(ts)
    pol = ts_processors.genotypes_and_distances({"polarised": True, "phased": True})
    pol_feat = pol(ts)
    assert not np.allclose(unpol_feat, pol_feat)
    assert unpol_feat[..., :-1].max() == 1
    assert pol_feat[..., :-1].max() == 1
    assert (unpol_feat[..., :-1].sum(axis=1) / ts.num_samples).max() <= 0.5
    assert (pol_feat[..., :-1].sum(axis=1) / ts.num_samples).max() <= 1.0
    
    # unphased
    unpol = ts_processors.genotypes_and_distances({"polarised": False, "phased": False})
    unpol_feat = unpol(ts)
    pol = ts_processors.genotypes_and_distances({"polarised": True, "phased": False})
    pol_feat = pol(ts)
    assert not np.allclose(unpol_feat, pol_feat)
    assert unpol_feat[..., :-1].max() == 2
    assert pol_feat[..., :-1].max() == 2
    assert (unpol_feat[..., :-1].sum(axis=1) / ts.num_samples).max() <= 0.5
    assert (pol_feat[..., :-1].sum(axis=1) / ts.num_samples).max() <= 1.0
    

def test_cnn_extract():
    """
    dinf is always polarising by minor allele frequency
    """

    one_pop = ts_simulators.VariablePopulationSize({})
    ts, theta = one_pop(10)

    with pytest.raises(ValueError):
        ts_processors.cnn_extract({"polarised": True})
    
    # phased
    unpol = ts_processors.cnn_extract({"phased": True})
    unpol_feat = unpol(ts)
    assert unpol_feat[0].max() == 1
    assert (unpol_feat[0].sum(axis=0) / ts.num_samples).max() <= 0.5
    
    # unphased
    unpol = ts_processors.cnn_extract({"phased": False})
    unpol_feat = unpol(ts)
    assert unpol_feat[0].max() == 2
    assert (unpol_feat[0].sum(axis=0) / ts.num_samples).max() <= 0.5

    
def test_SPIDNA_processor():

    one_pop = ts_simulators.VariablePopulationSize({})
    ts, theta = one_pop(10)

    # phased
    unpol = ts_processors.SPIDNA_processor({"polarised": False, "phased": True})
    unpol_feat = unpol(ts)
    pol = ts_processors.SPIDNA_processor({"polarised": True, "phased": True})
    pol_feat = pol(ts)
    assert not np.allclose(unpol_feat, pol_feat)
    assert unpol_feat[1:].max() == 1
    assert pol_feat[1:].max() == 1
    assert (unpol_feat[1:].sum(axis=0) / ts.num_samples).max() <= 0.5
    assert (pol_feat[1:].sum(axis=0) / ts.num_samples).max() <= 1.0
    
    # unphased
    unpol = ts_processors.SPIDNA_processor({"polarised": False, "phased": False})
    unpol_feat = unpol(ts)
    pol = ts_processors.SPIDNA_processor({"polarised": True, "phased": False})
    pol_feat = pol(ts)
    assert not np.allclose(unpol_feat, pol_feat)
    assert unpol_feat[1:].max() == 2
    assert pol_feat[1:].max() == 2
    assert (unpol_feat[1:].sum(axis=0) / ts.num_samples).max() <= 0.5
    assert (pol_feat[1:].sum(axis=0) / ts.num_samples).max() <= 1.0
    

def test_ReLERNN_processor():

    one_pop = ts_simulators.VariablePopulationSize({})
    ts, theta = one_pop(10)

    with pytest.raises(ValueError):
        ts_processors.ReLERNN_processor({"phased": False})

    # phased
    unpol = ts_processors.ReLERNN_processor({"polarised": False, "phased": True})
    unpol_feat = unpol(ts)
    pol = ts_processors.ReLERNN_processor({"polarised": True, "phased": True})
    pol_feat = pol(ts)
    assert not np.allclose(unpol_feat, pol_feat)
    assert unpol_feat[:, 1:].max() == 1
    assert pol_feat[:, 1:].max() == 1
    assert (unpol_feat[:, 1:].sum(axis=0) / ts.num_samples).max() <= 0.5
    assert (pol_feat[:, 1:].sum(axis=0) / ts.num_samples).max() <= 1.0
