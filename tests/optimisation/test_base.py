"""Tests for base optimization classes and types."""

import pytest

from backtester.optmisation.base import (
    BaseOptimization,
    OptimizationDirection,
    OptimizationMetadata,
    OptimizationType,
)


class TestOptimizationType:
    """Test cases for OptimizationType enum."""

    def test_optimization_types(self) -> None:
        """Test that all optimization types are defined."""
        assert OptimizationType.SINGLE_OBJECTIVE.value == "single_objective"
        assert OptimizationType.MULTI_OBJECTIVE.value == "multi_objective"
        assert OptimizationType.CONSTRAINED.value == "constrained"


class TestOptimizationDirection:
    """Test cases for OptimizationDirection enum."""

    def test_optimization_directions(self) -> None:
        """Test that all optimization directions are defined."""
        assert OptimizationDirection.MAXIMIZE.value == "maximize"
        assert OptimizationDirection.MINIMIZE.value == "minimize"


class TestOptimizationMetadata:
    """Test cases for OptimizationMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating optimization metadata."""
        metadata = OptimizationMetadata(
            study_name="test_study",
            optimization_type=OptimizationType.SINGLE_OBJECTIVE,
            direction=OptimizationDirection.MAXIMIZE,
            n_trials=100,
            start_time="2023-01-01T00:00:00",
            parameters_count=5,
        )
        assert metadata.study_name == "test_study"
        assert metadata.optimization_type == OptimizationType.SINGLE_OBJECTIVE
        assert metadata.direction == OptimizationDirection.MAXIMIZE
        assert metadata.n_trials == 100
        assert metadata.start_time == "2023-01-01T00:00:00"
        assert metadata.parameters_count == 5

    def test_metadata_optional_fields(self) -> None:
        """Test creating metadata with optional fields."""
        metadata = OptimizationMetadata(
            study_name="test_study",
            optimization_type=OptimizationType.SINGLE_OBJECTIVE,
            direction=OptimizationDirection.MAXIMIZE,
            n_trials=100,
            start_time="2023-01-01T00:00:00",
            parameters_count=5,
            dataset_info={"ticker": "SPY", "start_date": "2020-01-01"},
        )
        assert metadata.dataset_info == {"ticker": "SPY", "start_date": "2020-01-01"}


class TestBaseOptimization:
    """Test cases for BaseOptimization abstract class."""

    def test_initialize_base_optimization(self) -> None:
        """Test that BaseOptimization can be initialized."""
        # This is an abstract class, but we can test the initialization
        import logging

        logger = logging.getLogger(__name__)
        base_opt = BaseOptimization(logger)
        assert base_opt.logger == logger

    def test_validate_params_valid(self) -> None:
        """Test parameter validation with valid parameters."""
        base_opt = BaseOptimization()
        valid_params = {
            "leverage_base": 1.5,
            "leverage_alpha": 3.0,
            "ma_short": 5,
            "strategy_type": "moving_average",
        }
        assert base_opt.validate_params(valid_params) is True

    def test_validate_params_invalid_dict(self) -> None:
        """Test parameter validation with invalid type."""
        base_opt = BaseOptimization()
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            base_opt.validate_params("not_a_dict")

    def test_validate_params_invalid_key(self) -> None:
        """Test parameter validation with invalid key type."""
        base_opt = BaseOptimization()
        invalid_params = {123: "value"}  # Non-string key
        with pytest.raises(ValueError, match="Parameter key must be string"):
            base_opt.validate_params(invalid_params)

    def test_validate_params_invalid_value(self) -> None:
        """Test parameter validation with invalid value type."""
        base_opt = BaseOptimization()
        invalid_params = {"param": [1, 2, 3]}  # List value not in allowed types
        with pytest.raises(ValueError, match="Parameter value must be one of"):
            base_opt.validate_params(invalid_params)

    def test_get_optimization_info(self) -> None:
        """Test getting optimization info."""
        base_opt = BaseOptimization()
        # This will fail because get_study_name is not implemented
        with AttributeError:
            base_opt.get_optimization_info()
