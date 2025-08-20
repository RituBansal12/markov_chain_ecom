"""Unit tests for absorbing Markov chain analysis utilities."""
import pandas as pd
import numpy as np
from modelling_scripts.lib import analyze_absorbing_chain


def test_analyze_absorbing_chain():
    # States: START, VIEW, ADD, PURCHASE, DROP
    states = ["START", "VIEW", "ADD_TO_CART", "PURCHASE", "DROP"]
    # Build a simple matrix consistent with absorption
    tm = pd.DataFrame(0.0, index=states, columns=states)
    tm.loc["PURCHASE", :] = 0.0
    tm.loc["PURCHASE", "PURCHASE"] = 1.0
    tm.loc["DROP", :] = 0.0
    tm.loc["DROP", "DROP"] = 1.0
    tm.loc["START", "VIEW"] = 1.0
    tm.loc["VIEW", "ADD_TO_CART"] = 0.5
    tm.loc["VIEW", "DROP"] = 0.5
    tm.loc["ADD_TO_CART", "PURCHASE"] = 1.0

    res = analyze_absorbing_chain(tm, ["PURCHASE", "DROP"])
    B = res["B"]
    t = res["t"]

    # From START, purchase probability should be 0.5 (VIEW->ADD->PURCHASE path is 0.5)
    assert np.isclose(B.loc["START", "PURCHASE"], 0.5)
    assert np.isclose(B.loc["START", "DROP"], 0.5)
    # Expected steps finite and positive
    assert (t.values > 0).all()
