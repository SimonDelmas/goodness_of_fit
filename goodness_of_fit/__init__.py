"""
    goodness_of_fit: Function set for goodness of fit measure between two signals.
    ================================================
    Documentation is available in the docstrings

    Contents
    --------
    goodness_of_fit require numpy.
    The package provides the following measure :
    - Mean Error : me()
    - Mean Absolute Error : mae()
    - Root Mean Square Error : rmse()
    - Normalized Root Mean Square Error : nrmse()
    - Pearson product-moment correlation coefficient : r_pearson()
    - Coefficient of Determination : r2()
    - Index of Agreement : d()
    - Modified Index of Agreement : md()
    - Relative Index of Agreement : rd()
    - Ratio of Standard Deviations : rSD
    - Nash-sutcliffe Efficiency : nse()
    - Modified Nash-sutcliffe Efficiency : mnse()
    - Relative Nash-sutcliffe Efficiency : rnse()
    - Kling Gupta Efficiency : kge()
    - Deviation of gain : dg()
    - Standard deviation of residual : sdr()

    All available function are also provided through the dictionary : gof_measure

    It also provide an helper :
    is_flat : check if a signal is flat or not

    Utility tools
    -------------
     __version__ : goodness_of_fit version string
"""

import warnings
import numpy as np

__version__ = "1.0.1"


def _check_inputs(cal, obs):
    """
    Helper to check input data.

    """
    cal = np.asarray(cal)
    obs = np.asarray(obs)

    are_comparable = cal.shape == obs.shape and cal.ndim == obs.ndim == 1
    if not are_comparable:
        raise ValueError("Arguments must be 1D numpy.ndarrays of the same shape!")

    return cal, obs


def is_flat(signal, eps=1e-2):
    """
    Return True if the signal is flat.
    The signal is consider flat if the ration between standard deviation and mean of the signal is lower than a given threshold (eps).

    :param signal: (N,) array_like.
    :param eps: Tolerance criterion.
    :return: Bool.
    """
    s = np.asarray(signal)
    return np.std(s) < eps * np.mean(s)


def me(cal, obs):
    """
    Mean Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Mean Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'

    .. math::

        r = \\langle C \\rangle - \\langle O \\rangle

    """
    cal, obs = _check_inputs(cal, obs)
    result = np.mean(cal) - np.mean(obs)
    return result


def mae(cal, obs):
    """
    Mean Absolute Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Mean Absolute Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'

    .. math::

            r = \\langle \\vert C - O \\vert \\rangle

    """
    cal, obs = _check_inputs(cal, obs)
    result = np.mean(np.abs(cal - obs))
    return result


def rmse(cal, obs):
    """
    Root Mean Square Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Root Mean Square Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'

    .. math::

        r = \\langle \\vert (C - O) \\vert \\rangle

    """
    cal, obs = _check_inputs(cal, obs)
    result = np.sqrt(np.mean(np.power(cal - obs, 2.0)))
    return result


def nrmse(cal, obs, norm="std"):
    """
    Normalized Root Mean Square Error between 'cal' and 'obs', in the same units of 'cal' and 'obs'.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :param norm: Indicate the function to be used to normalise the RMS (std or maxmin)
    :return: Normalized Root Mean Square Error between 'cal' and 'obs'.

    .. math::

        r = \\frac{\\langle \\vert C - O \\vert \\rangle}{std(O)}

        r = \\frac{\\langle \\vert C - O \\vert \\rangle}{maxmin(O)}

    With :

    .. math::

        std(O)  = \\sqrt{\\frac{1}{N} \\sum (O_i - m_O)^2}

        maxmin(O) = max(O) - min(O)

    """
    cal, obs = _check_inputs(cal, obs)

    if norm == "std":
        cte = np.std(obs)
        if cte == 0.0:
            warnings.warn(
                "Normalisation parameter : standard deviation of observation is 0. Return NaN."
            )
            return np.nan
    elif norm == "maxmin":
        cte = np.max(obs) - np.min(obs)
        if cte == 0.0:
            warnings.warn(
                "Normalisation parameter : max(obs) - min(obs) is 0. Return NaN."
            )
            return np.nan
    else:
        raise ValueError("norm should be std or maxmin!")

    return rmse(cal, obs) / cte


def r_pearson(cal, obs):
    """
    The Pearson product-moment correlation coefficient (ranges from -1 to 1).
    Implementation freely inspired from scipy stats module : https://docs.scipy.org/doc/scipy/reference/stats.html

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Pearson product-moment correlation coefficient between cal and obs

    .. math::

        r = \\frac{\\sum (C_i - m_C) (O_i - m_O)}{\\sqrt{\\sum (C_i - m_C)^2 \\sum (O_i - m_O)^2}}

    where :math:`m_C` is the mean of the vector :math:`C` and :math:`m_O` is the mean of the vector :math:`O`.

    """
    cal, obs = _check_inputs(cal, obs)

    xm, ym = cal - cal.mean(), obs - obs.mean()
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if r_den == 0.0:
        warnings.warn("Both signal are flat and null. Return Nan.")
        return np.nan
    return r_num / r_den


# def r_spearman(cal, obs):
#     """
#     The Spearman coefficient (ranges from -1 to 1).
#
#     :param cal: (N,) array_like of calculated values
#     :param obs: (N,) array_like of observed values
#     :return: Scalar value Spearman correlation coefficient between cal and obs
#     """
#     cal, obs = _check_inputs(cal, obs)
#     result = stats.spearmanr(cal, obs)
#     return result[0]


def r2(cal, obs):
    """
    'R2' is the Coefficient of Determination
    The coefficient of determination is such that 0 <  R2 < 1,  and denotes the strength 
    of the linear association between cal and obs.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Coefficient of Determination between cal and obs

    .. math::

        r = r_{pearson}^2

    """
    result = r_pearson(cal, obs) ** 2.0
    return result


def d(cal, obs):
    """
    Index of Agreement (Willmott et al., 1984) range from 0.0 to 1.0 
    and the closer to 1 the better the performance of the model.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value index of Agreement between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum (O_i - C_i)^2}
                      {\\sum (\\vert C_i - m_O \\vert + \\vert O_i - m_O \\vert)^2}

    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    """
    return md(cal, obs, order=2)


def md(cal, obs, order=1):
    """
    Modify Index of Agreement (Willmott et al., 1984) range from 0.0 to 1.0 
    and the closer to 1 the better the performance of the model.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :param order: exponent to be used in the computation. Default is 1
    :return: Modified Index of Agreement between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum (O_i - C_i)^n}
                      {\\sum (\\vert C_i - m_O \\vert + \\vert O_i - m_O \\vert)^n}

    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    """
    cal, obs = _check_inputs(cal, obs)

    obs_mean = obs.mean()
    denominator = np.sum(
        np.power(np.abs(cal - obs_mean) + np.abs(obs - obs_mean), order)
    )
    if denominator == 0.0:
        warnings.warn("Index of agreement potential error is null! Return NaN.")
        return np.nan

    nominator = np.sum(np.abs(np.power(obs - cal, order)))
    return 1 - (nominator / denominator)


def rd(cal, obs):
    """
    Relative Index of Agreement (Willmott et al., 1984) range from 0.0 to 1.0 
    and the closer to 1 the better the performance of the model.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Relative Index of Agreement between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum \\left( \\frac{O_i - C_i}{O_i} \\right)^2}
                      {\\sum \\left( \\frac{\\vert C_i - m_O \\vert + \\vert O_i - m_O \\vert}{O_i} \\right)^2}

    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    """
    cal, obs = _check_inputs(cal, obs)

    if not np.count_nonzero(obs) > 0:
        warnings.warn(
            "Cannot compute relative index of agreement as there is zero value in observation!"
        )
        return np.nan

    obs_mean = obs.mean()
    if obs_mean == 0.0:
        warnings.warn(
            "Cannot compute relative index of agreement as mean of observation is null!"
        )
        return np.nan

    denominator = np.sum(
        np.power((np.abs(cal - obs_mean) + np.abs(obs - obs_mean)) / obs_mean, 2.0)
    )
    if denominator == 0.0:
        warnings.warn(
            "Relative index of agreement potential error is null! Return NaN."
        )
        return np.nan

    nominator = np.sum(np.power((obs - cal) / obs, 2.0))
    return 1 - (nominator / denominator)


def rsd(cal, obs):
    """
    Ratio of Standard Deviations.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Ratio of Standard Deviations between 'cal' and 'obs'

    .. math::

        r = std(C) / std(O)

    """
    cal, obs = _check_inputs(cal, obs)
    denominator = np.std(obs)
    if denominator > 0.0:
        return np.std(cal) / denominator
    else:
        warnings.warn("Standard deviation of observation is zero! Return NaN.")
        return np.nan


def nse(cal, obs):
    """
    Nash-Sutcliffe efficiencies (Nash and Sutcliffe, 1970) range from -Inf to 1. 
    An efficiency of 1 (NSE = 1) corresponds to a perfect match of modeled to the observed data
    as the mean of the observed data, whereas an efficiency less than zero (-Inf < NSE < 0) 
    occurs when the observed mean is a better predictor than the model.
    Essentially, the closer the model efficiency is to 1, the more accurate the model is.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Nash-Sutcliffe Efficiency between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum \\left( O_i - C_i \\right)^2 }{\\sum \\left( O_i - m_O \\right)^2}

    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    """
    cal, obs = _check_inputs(cal, obs)
    obs_mean = obs.mean()
    denominator = np.sum(np.power(obs - obs_mean, 2.0))
    if denominator == 0.0:
        warnings.warn("Observation variance is null. Return NaN.")
        return np.nan
    return 1.0 - np.sum(np.power(obs - cal, 2.0)) / denominator


def mnse(cal, obs, order=1):
    """
    Modify Nash-sutcliffe Efficiency
    Nash-Sutcliffe efficiency not "inflated" by squared values.
    Essentially, the closer the model efficiency is to 1, the more accurate the model is.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :param order: exponent to be used in the computation. Default is 1
    :return: Scalar value Modified Nash-sutcliffe Efficiency between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum \\left( O_i - C_i \\right)^n }{\\sum \\left( O_i - m_O \\right)^n}

    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    """
    cal, obs = _check_inputs(cal, obs)
    obs_mean = obs.mean()
    denominator = np.sum(np.abs(np.power(obs - obs_mean, order)))
    if denominator == 0.0:
        warnings.warn("Observation variance is null. Return NaN.")
        return np.nan
    return 1.0 - np.sum(np.abs(np.power(obs - cal, order))) / denominator


def rnse(cal, obs):
    """
    Relative Nash-sutcliffe Efficiency
    Essentially, the closer the model efficiency is to 1, the more accurate the model is.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Modified Nash-sutcliffe Efficiency between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum \\frac{\\left( O_i - C_i \\right)^n}{O_i} }{ \\frac{\\sum \\left( O_i - m_O \\right)^2}{m_O} }

    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    """
    cal, obs = _check_inputs(cal, obs)

    if not np.count_nonzero(obs) > 0:
        warnings.warn(
            "Cannot compute relative efficiency as there is zero value in observation!"
        )
        return np.nan

    obs_mean = obs.mean()
    if obs_mean == 0.0:
        warnings.warn(
            "Cannot compute relative efficiency as mean of observation is null!"
        )
        return np.nan

    denominator = np.sum(np.power((obs - obs_mean) / obs_mean, 2.0))
    if denominator == 0.0:
        warnings.warn("Normalized variance of observation is null! Return NaN.")
        return np.nan

    return 1.0 - np.sum(np.power((obs - cal) / obs, 2.0)) / denominator


def kge(cal, obs):
    """
    Kling Gupta Efficiency. This measure was developed by Gupta et al. (2009) to provide a diagnostically
    interesting decomposition of the Nash-Sutcliffe efficiency (and hence MSE), which facilitates the
    analysis of the relative importance of its different components (correlation, bias and variability) in
    the context of hydrological modelling.
    Essentially, the closer the model efficiency is to 1, the more accurate the model is.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Kling Gupta Efficiency between 'cal' and 'obs'

    .. math::

        r = 1 - \\sqrt{ \\left( r_{pearson} - 1 \\right)^2 + \\left( \\frac{std(C)}{std(O)} - 1 \\right)^2 + \\left( \\frac{m_C}{m_O} - 1 \\right)^2}


    where :math:`m_O` is the mean of the vector :math:`O` of observed values, and :math:`m_C` is the mean of the vector :math:`C` of calculated values.

    """
    cal, obs = _check_inputs(cal, obs)

    o_std = np.std(obs)
    if o_std == 0.0:
        return np.nan

    o_mean = np.mean(obs)
    if o_mean == 0.0:
        return np.nan

    std_ratio = np.std(cal) / o_std
    mean_ratio = np.mean(cal) / o_mean
    pears = r_pearson(cal, obs)
    return 1.0 - np.sqrt(
        (pears - 1.0) ** 2.0 + (std_ratio - 1.0) ** 2.0 + (mean_ratio - 1.0) ** 2.0
    )


def dg(cal, obs):
    """
    Deviation of gain. Vary between 0 and 1.
    Essentially, the closer the deviation is to 0, the more accurate the model is.

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Scalar value Deviation of Gain between 'cal' and 'obs'

    .. math::

        r = 1 - \\frac{\\sum (O_i - C_i)^2}{\\sum (O_i - m_O)}

    where :math:`m_O` is the mean of the vector :math:`O`.

    """
    cal, obs = _check_inputs(cal, obs)

    den = np.sum((obs - obs.mean()) ** 2.0)
    if den == 0.0:
        return np.nan

    return 1.0 - np.sum((obs - cal) ** 2.0) / den


def sdr(cal, obs):
    """
    Standard deviation of residual :

    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :return: Standard deviation of residual between 'cal' and 'obs'

    .. math::

        r = \\sqrt{ \\langle \\left[ (cal - obs) - \\langle cal \\rangle + \\langle obs \\rangle \\right]^2 \\rangle}
    """
    cal, obs = _check_inputs(cal, obs)

    return np.sqrt(np.mean(np.power((cal - obs) - cal.mean() + obs.mean(), 2.0)))


gof_measure = {
    "me": me,
    "mae": mae,
    "rmse": rmse,
    "nrmse": nrmse,
    "r_pearson": r_pearson,
    "r2": r2,
    "d": d,
    "md": md,
    "rd": rd,
    "rSD": rsd,
    "NSE": nse,
    "mNSE": mnse,
    "rNSE": rnse,
    "KGE": kge,
    "DG": dg,
    "sdr": sdr,
}
