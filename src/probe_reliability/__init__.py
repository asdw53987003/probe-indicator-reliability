# -*- coding: utf-8 -*-
"""probe_reliability: split-half check of probe-vehicle indicators for crash-frequency models and site lists.

Reference implementation of Algorithm A.1 of "Are probe vehicle data enough? Indicator reliability for freeway
crash-frequency models and site screening".
"""
from .calibration import Calibration, calibration_slope, slope_heterogeneity
from .correction import SimexResult, simex_nb2, unit_error_variances
from .executed import executed_check, held_out_reliability
from .indicators import (INDICATORS, ehc, hdc, indicators_table, standard_columns, sums_from_points,
                         thin_records, unit_indicators)
from .lists import capture, list_overlap, predicted_overlap, ranking_reliability, top_set
from .nb2 import NB2Fit, fit_nb2
from .requirements import k_hat, n_required, reliability_at, volume_summary, weeks_needed
from .split_half import SplitHalfResult, attenuation_matrix, split_half_check

__version__ = '0.1.0'

__all__ = ['Calibration', 'calibration_slope', 'slope_heterogeneity', 'SimexResult', 'simex_nb2',
           'unit_error_variances', 'executed_check', 'held_out_reliability', 'INDICATORS', 'ehc', 'hdc',
           'indicators_table', 'standard_columns', 'sums_from_points', 'thin_records', 'unit_indicators', 'capture',
           'list_overlap', 'predicted_overlap', 'ranking_reliability', 'top_set', 'NB2Fit', 'fit_nb2', 'k_hat',
           'n_required', 'reliability_at', 'volume_summary', 'weeks_needed', 'SplitHalfResult', 'attenuation_matrix',
           'split_half_check']
