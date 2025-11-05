"""Optuna Optimization Module for the Backtester.

This module provides comprehensive optimization functionality using Optuna
for hyperparameter tuning and strategy optimization.
"""

from .base import BaseOptimization, OptimizationDirection, OptimizationType
from .objective import ObjectiveResult, OptimizationObjective
from .parameter_space import OptimizationConfig, ParameterSpace
from .results import ResultsAnalyzer
from .runner import OptimizationResult, OptimizationRunner
from .study_manager import OptunaStudyManager

__all__ = [
    'OptunaStudyManager',
    'OptimizationObjective',
    'ObjectiveResult',
    'ParameterSpace',
    'OptimizationConfig',
    'OptimizationRunner',
    'OptimizationResult',
    'ResultsAnalyzer',
    'BaseOptimization',
    'OptimizationType',
    'OptimizationDirection',
]
