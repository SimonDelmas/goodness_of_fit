import pytest
from pathlib import Path
import numpy as np
from goodness_of_fit import *


@pytest.fixture(scope="module")
def path_data_folder():
    return Path(__file__).parent / 'data'


@pytest.fixture(scope="module")
def observed_data(path_data_folder):
    return np.genfromtxt(str(path_data_folder / 'obs.csv'), delimiter=',')


@pytest.fixture(scope="module")
def calculated_data(path_data_folder):
    return np.genfromtxt(str(path_data_folder / 'cal.csv'), delimiter=',')


def test_me(observed_data, calculated_data):
    res = me(calculated_data, calculated_data)
    assert res == 0.0

    res = me(calculated_data, observed_data)
    assert res == pytest.approx(-0.04539888888888921)


def test_mae(observed_data, calculated_data):
    res = mae(calculated_data, calculated_data)
    assert res == 0.0

    res = mae(calculated_data, observed_data)
    assert res == pytest.approx(0.11492975308641971)


def test_rmse(observed_data, calculated_data):
    res = rmse(calculated_data, calculated_data)
    assert res == 0.0

    res = rmse(calculated_data, observed_data)
    assert res == pytest.approx(0.15444431323335747)


def test_nrmse(observed_data, calculated_data):
    res = nrmse(calculated_data, calculated_data)
    assert res == 0.0

    res = nrmse(calculated_data, observed_data)
    assert res == pytest.approx(0.922292762943825)


def test_pearson_correlation(observed_data, calculated_data):
    res = r_pearson(calculated_data, calculated_data)
    assert res == 1.0

    res = r_pearson(calculated_data, observed_data)
    assert res == pytest.approx(0.5500135621782686)


def test_r2(observed_data, calculated_data):
    res = r2(calculated_data, calculated_data)
    assert res == 1.0

    res = r2(calculated_data, observed_data)
    assert res == pytest.approx(0.3025149185800281)


def test_index_of_agreement(observed_data, calculated_data):
    res = d(calculated_data, calculated_data)
    assert res == 1.0

    res = d(calculated_data, observed_data)
    assert res == pytest.approx(0.7255005827087879)


def test_relative_index_of_agreement(observed_data, calculated_data):
    res = rd(calculated_data, calculated_data)
    assert res == 1.0

    res = rd(calculated_data, observed_data)
    assert res == pytest.approx(0.7355506542041386)


def test_rsd(observed_data, calculated_data):
    res = rsd(calculated_data, calculated_data)
    assert res == 1.0

    res = rsd(calculated_data, observed_data)
    assert res == pytest.approx(0.8322180792361873)


def test_nse(observed_data, calculated_data):
    res = nse(calculated_data, calculated_data)
    assert res == 1.0

    res = nse(calculated_data, observed_data)
    assert res == pytest.approx(0.14937605942144527)


def test_mnse(observed_data, calculated_data):
    res = mnse(calculated_data, calculated_data)
    assert res == 1.0

    res = mnse(calculated_data, observed_data)
    assert res == pytest.approx(0.03637720441650172)


def test_rnse(observed_data, calculated_data):
    res = rnse(calculated_data, calculated_data)
    assert res == 1.0

    res = rnse(calculated_data, observed_data)
    assert res == pytest.approx(0.18051940938856725)


def test_kge(observed_data, calculated_data):
    res = kge(calculated_data, calculated_data)
    assert res == 1.0

    res = kge(calculated_data, observed_data)
    assert res == pytest.approx(0.5197011486544534)


def test_dg(observed_data, calculated_data):
    res = dg(calculated_data, calculated_data)
    assert res == 1.0

    res = dg(calculated_data, observed_data)
    assert res == pytest.approx(0.14937605942144527)


def test_sdr(observed_data, calculated_data):
    res = sdr(calculated_data, calculated_data)
    assert res == 0.0

    res = sdr(calculated_data, observed_data)
    assert res == pytest.approx(0.14762109191364817)

