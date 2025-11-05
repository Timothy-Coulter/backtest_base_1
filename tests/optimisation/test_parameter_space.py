"""Tests for parameter space and optimization configuration."""

import pytest

from backtester.optmisation.parameter_space import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
)


class TestParameterDefinition:
    """Test cases for ParameterDefinition class."""

    def test_float_parameter_creation(self) -> None:
        """Test creating float parameter definition."""
        param = ParameterDefinition(
            name="test_param",
            param_type="float",
            low=0.0,
            high=1.0,
            step=0.1,
        )
        assert param.name == "test_param"
        assert param.param_type == "float"
        assert param.low == 0.0
        assert param.high == 1.0
        assert param.step == 0.1

    def test_int_parameter_creation(self) -> None:
        """Test creating int parameter definition."""
        param = ParameterDefinition(
            name="test_param",
            param_type="int",
            low=1,
            high=10,
            step=1,
        )
        assert param.name == "test_param"
        assert param.param_type == "int"
        assert param.low == 1
        assert param.high == 10
        assert param.step == 1

    def test_categorical_parameter_creation(self) -> None:
        """Test creating categorical parameter definition."""
        param = ParameterDefinition(
            name="test_param",
            param_type="categorical",
            choices=["a", "b", "c"],
        )
        assert param.name == "test_param"
        assert param.param_type == "categorical"
        assert param.choices == ["a", "b", "c"]

    def test_invalid_parameter_type(self) -> None:
        """Test that invalid parameter type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid parameter type"):
            ParameterDefinition(
                name="test_param",
                param_type="invalid_type",
                low=0.0,
                high=1.0,
            )

    def test_missing_bounds_for_float(self) -> None:
        """Test that missing bounds for float parameter raises ValueError."""
        with pytest.raises(ValueError, match="Low and high bounds required"):
            ParameterDefinition(
                name="test_param",
                param_type="float",
                low=None,
                high=None,
            )

    def test_invalid_low_high_order(self) -> None:
        """Test that low >= high raises ValueError."""
        with pytest.raises(ValueError, match="Low must be less than high"):
            ParameterDefinition(
                name="test_param",
                param_type="float",
                low=1.0,
                high=0.0,
            )


class TestParameterSpace:
    """Test cases for ParameterSpace class."""

    def test_initialization(self) -> None:
        """Test ParameterSpace initialization."""
        space = ParameterSpace()
        assert space.get_parameter_count() == 0

    def test_add_float_parameter(self) -> None:
        """Test adding float parameter."""
        space = ParameterSpace()
        space.add_float("leverage_base", 1.0, 10.0, step=0.5)
        assert space.get_parameter_count() == 1
        assert "leverage_base" in space.get_parameter_names()

    def test_add_int_parameter(self) -> None:
        """Test adding int parameter."""
        space = ParameterSpace()
        space.add_int("ma_short", 1, 50, step=1)
        assert space.get_parameter_count() == 1
        assert "ma_short" in space.get_parameter_names()

    def test_add_categorical_parameter(self) -> None:
        """Test adding categorical parameter."""
        space = ParameterSpace()
        space.add_categorical("strategy_type", ["ma", "momentum", "mean_reversion"])
        assert space.get_parameter_count() == 1
        assert "strategy_type" in space.get_parameter_names()

    def test_add_loguniform_parameter(self) -> None:
        """Test adding loguniform parameter."""
        space = ParameterSpace()
        space.add_loguniform("learning_rate", 0.001, 0.1, q=0.001)
        assert space.get_parameter_count() == 1
        assert "learning_rate" in space.get_parameter_names()

    def test_add_loguniform_invalid_low(self) -> None:
        """Test that loguniform with non-positive low bound raises ValueError."""
        space = ParameterSpace()
        with pytest.raises(ValueError, match="positive lower bound"):
            space.add_loguniform("test_param", -1.0, 10.0)

    def test_method_chaining(self) -> None:
        """Test method chaining for parameter addition."""
        space = (
            ParameterSpace()
            .add_float("param1", 0.0, 1.0)
            .add_int("param2", 1, 10)
            .add_categorical("param3", ["a", "b"])
        )
        assert space.get_parameter_count() == 3

    def test_create_grid_space(self) -> None:
        """Test creating grid space from parameter definitions."""
        space = (
            ParameterSpace()
            .add_float("float_param", 0.0, 1.0, step=0.5)
            .add_int("int_param", 1, 5)
            .add_categorical("cat_param", ["a", "b"])
        )
        grid_space = space.create_grid_space()

        assert "float_param" in grid_space
        assert "int_param" in grid_space
        assert "cat_param" in grid_space
        assert len(grid_space["float_param"]) == 3  # 0.0, 0.5, 1.0
        assert len(grid_space["int_param"]) == 4  # 1, 2, 3, 4
        assert grid_space["cat_param"] == ["a", "b"]


class TestOptimizationConfig:
    """Test cases for OptimizationConfig class."""

    def test_initialization(self) -> None:
        """Test OptimizationConfig initialization."""
        config = OptimizationConfig()
        assert config.n_trials == 100
        assert config.timeout is None
        assert config.n_jobs == 1
        assert config.show_progress_bar is True
        assert config.direction == "maximize"
        assert config.study_name is None

    def test_set_trials(self) -> None:
        """Test setting number of trials."""
        config = OptimizationConfig()
        config.set_trials(200)
        assert config.n_trials == 200

    def test_set_timeout(self) -> None:
        """Test setting timeout."""
        config = OptimizationConfig()
        config.set_timeout(300)
        assert config.timeout == 300

    def test_set_parallel_jobs(self) -> None:
        """Test setting number of parallel jobs."""
        config = OptimizationConfig()
        config.set_parallel_jobs(4)
        assert config.n_jobs == 4

    def test_set_sampler(self) -> None:
        """Test setting sampler configuration."""
        config = OptimizationConfig()
        config.set_sampler("tpe", seed=42)
        assert config.sampler_name == "tpe"
        assert config.sampler_kwargs == {"seed": 42}

    def test_set_storage(self) -> None:
        """Test setting storage configuration."""
        config = OptimizationConfig()
        config.set_storage("postgresql://localhost/db")
        assert config.storage_url == "postgresql://localhost/db"

    def test_set_study_name(self) -> None:
        """Test setting study name."""
        config = OptimizationConfig()
        config.set_study_name("test_study")
        assert config.study_name == "test_study"

    def test_set_direction(self) -> None:
        """Test setting optimization direction."""
        config = OptimizationConfig()
        config.set_direction("minimize")
        assert config.direction == "minimize"

    def test_set_invalid_direction(self) -> None:
        """Test setting invalid direction raises ValueError."""
        config = OptimizationConfig()
        with pytest.raises(ValueError, match="Direction must be"):
            config.set_direction("invalid_direction")

    def test_get_sampler(self) -> None:
        """Test getting configured sampler."""
        config = OptimizationConfig()
        config.set_sampler("random", seed=42)
        sampler = config.get_sampler()
        assert sampler is not None

    def test_method_chaining(self) -> None:
        """Test method chaining for configuration."""
        config = (
            OptimizationConfig()
            .set_trials(150)
            .set_timeout(600)
            .set_parallel_jobs(2)
            .set_direction("minimize")
            .set_study_name("test_study")
        )
        assert config.n_trials == 150
        assert config.timeout == 600
        assert config.n_jobs == 2
        assert config.direction == "minimize"
        assert config.study_name == "test_study"
