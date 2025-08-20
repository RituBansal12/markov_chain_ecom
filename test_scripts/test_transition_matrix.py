"""Unit tests for building transition matrices from dummy transitions."""
import pandas as pd
import numpy as np
from modelling_scripts.lib import build_transition_matrix_from_transitions


def test_build_transition_matrix_simple():
    # Dummy transitions: START->VIEW->ADD->PURCHASE and START->VIEW->DROP
    df = pd.DataFrame([
        {"visitorid": 1, "session_id": 1, "from_state": "START", "to_state": "VIEW"},
        {"visitorid": 1, "session_id": 1, "from_state": "VIEW", "to_state": "ADD_TO_CART"},
        {"visitorid": 1, "session_id": 1, "from_state": "ADD_TO_CART", "to_state": "PURCHASE"},
        {"visitorid": 2, "session_id": 1, "from_state": "START", "to_state": "VIEW"},
        {"visitorid": 2, "session_id": 1, "from_state": "VIEW", "to_state": "DROP"},
    ])
    tm = build_transition_matrix_from_transitions(df)

    # Rows sum to 1
    assert np.allclose(tm.sum(axis=1).values, 1.0)
    # Absorbing rows
    assert tm.loc["PURCHASE", "PURCHASE"] == 1.0
    assert tm.loc["DROP", "DROP"] == 1.0
    # Probabilities
    # From START -> VIEW must be 1
    assert tm.loc["START", "VIEW"] == 1.0
    # From VIEW -> ADD vs DROP: ADD 1/2, DROP 1/2
    assert np.isclose(tm.loc["VIEW", "ADD_TO_CART"], 0.5)
    assert np.isclose(tm.loc["VIEW", "DROP"], 0.5)
