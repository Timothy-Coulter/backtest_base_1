"""Integration test for Optuna optimization workflow with BacktestEngine.

This test demonstrates the complete optimization pipeline including:
- Parameter space definition
- Study management
- Objective function execution
- Results analysis
- Different optimization types (single, multi, constrained)
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from backtester.optmisation.objective import create_optimization_objective
from backtester.optmisation.parameter_space import OptimizationConfig, ParameterSpace
from backtester.optmisation.runner import OptimizationRunner


class MockBacktestEngine:
    """Mock BacktestEngine for integration testing."""

    def __init__(self) -> None:
        """Initialize the mock backtest engine."""
        self.call_count = 0

    def load_data(
        self, ticker: str, start_date: str, end_date: str, interval: str = "1mo"
    ) -> pd.DataFrame:
        """Mock data loading."""
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        np.random.seed(42)  # For reproducible tests
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, len(dates)),
                "High": np.random.uniform(100, 200, len(dates)),
                "Low": np.random.uniform(100, 200, len(dates)),
                "Close": np.random.uniform(100, 200, len(dates)),
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )
        return data

    def run_backtest(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1mo",
        strategy_params: dict = None,
        portfolio_params: dict = None,
    ) -> dict:
        """Simulate running backtest with given parameters."""
        self.call_count += 1
        strategy_params = strategy_params or {}
        portfolio_params = portfolio_params or {}

        # Simulate realistic backtest results based on parameters
        base_return = 0.08 + (strategy_params.get("leverage_base", 2.0) - 1.0) * 0.05
        ma_bonus = strategy_params.get("ma_short", 10) / 100.0 * 0.02
        alpha_bonus = strategy_params.get("leverage_alpha", 3.0) / 10.0 * 0.03

        total_return = base_return + ma_bonus + alpha_bonus + np.random.normal(0, 0.01)

        # Calculate realistic metrics
        sharpe_ratio = total_return / 0.15 if total_return != 0 else 0  # Simplified calculation
        sortino_ratio = sharpe_ratio * 1.1
        max_drawdown = -abs(total_return) * 0.5 + np.random.normal(0, 0.01)
        trades_count = int(
            50 + strategy_params.get("ma_short", 10) * 2 + np.random.randint(-10, 11)
        )

        return {
            "performance": {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "trades_count": trades_count,
                "volatility": abs(max_drawdown) * 2,
                "win_rate": 0.55 + np.random.normal(0, 0.05),
            }
        }


class TestOptimizationIntegration:
    """Integration tests for complete optimization workflow."""

    @pytest.fixture
    def temp_db_path(self) -> str:
        """Create temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(temp_path)

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger for testing."""
        return Mock()

    @pytest.fixture
    def mock_backtest_engine(self):
        """Create mock backtest engine for testing."""
        return MockBacktestEngine()

    @pytest.fixture
    def parameter_space(self) -> ParameterSpace:
        """Create parameter space for testing."""
        space = ParameterSpace()

        # Define realistic parameters for moving average strategy
        space.define_parameter(
            "leverage_base",
            "float",
            low=1.0,
            high=3.0,
            default=2.0,
            step=0.1,
        )

        space.define_parameter(
            "leverage_alpha",
            "float",
            low=1.0,
            high=5.0,
            default=3.0,
            step=0.2,
        )

        space.define_parameter(
            "ma_short",
            "int",
            low=5,
            high=30,
            default=10,
        )

        space.define_parameter(
            "ma_long",
            "int",
            low=20,
            high=100,
            default=50,
        )

        return space

    def test_single_objective_optimization_workflow(
        self, temp_db_path, mock_logger, mock_backtest_engine, parameter_space
    ) -> None:
        """Test complete single-objective optimization workflow."""
        # Setup configuration
        config = OptimizationConfig(
            study_name=f"single_obj_test_{int(datetime.now().timestamp())}",
            optimization_type="single_objective",
            objective="total_return",
            n_trials=10,  # Small number for fast testing
            storage_url=f"sqlite:///{temp_db_path}",
            direction="maximize",
            n_jobs=1,
            seed=42,
        )

        # Create objective
        objective = create_optimization_objective(
            backtest_engine=mock_backtest_engine,
            ticker="TEST",
            start_date="2020-01-01",
            end_date="2020-12-31",
            objective_type="single",
            logger=mock_logger,
        )

        # Create optimization runner
        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=parameter_space,
            objective=objective,
            config=config,
            logger=mock_logger,
        )

        # Run optimization
        result = runner.optimize(n_trials=10, show_progress_bar=False)

        # Verify results structure
        assert result is not None
        assert result.best_params is not None
        assert isinstance(result.best_params, dict)
        assert result.best_value is not None
        assert isinstance(result.best_value, float)
        assert result.n_trials == 10
        assert result.optimization_time > 0

        # Verify best parameters are valid and within bounds
        assert 1.0 <= result.best_params["leverage_base"] <= 3.0
        assert 5 <= result.best_params["ma_short"] <= 30
        assert 20 <= result.best_params["ma_long"] <= 100
        assert 1.0 <= result.best_params["leverage_alpha"] <= 5.0

        # Verify optimization value is reasonable
        assert isinstance(result.best_value, float)
        assert -1.0 <= result.best_value <= 1.0  # Reasonable return bounds

        # Verify dictionary conversion
        result_dict = result.to_dict()
        assert "best_params" in result_dict
        assert "best_value" in result_dict
        assert "n_trials" in result_dict

    def test_objective_function_integration(self, mock_logger, mock_backtest_engine) -> None:
        """Test objective function integration with mock engine."""
        # Create objective
        objective = create_optimization_objective(
            backtest_engine=mock_backtest_engine,
            ticker="TEST",
            start_date="2020-01-01",
            end_date="2020-12-31",
            objective_type="single",
            logger=mock_logger,
        )

        # Test parameters
        test_params = {
            "leverage_base": 2.0,
            "leverage_alpha": 3.0,
            "ma_short": 10,
            "ma_long": 50,
        }

        # Test single objective
        result = objective._run_backtest(test_params)
        assert result.value is not None
        assert "sharpe_ratio" in result.metrics
        assert "total_return" in result.metrics

    def test_parameter_space_validation(self, parameter_space, mock_logger) -> None:
        """Test that parameter spaces are properly validated."""
        # Test parameter space structure
        assert parameter_space._parameters is not None

        # Verify all parameters have required attributes
        for _param_name, param_def in parameter_space._parameters.items():
            assert "type" in param_def
            assert "low" in param_def
            assert "high" in param_def
            assert "default" in param_def

            assert param_def["low"] < param_def["high"]
            assert param_def["default"] >= param_def["low"]
            assert param_def["default"] <= param_def["high"]

    def test_multi_objective_optimization_workflow(
        self, temp_db_path, mock_logger, mock_backtest_engine, parameter_space
    ) -> None:
        """Test complete multi-objective optimization workflow."""
        # Setup configuration for multi-objective optimization
        config = OptimizationConfig(
            study_name=f"multi_obj_test_{int(datetime.now().timestamp())}",
            optimization_type="multi_objective",
            objective=["total_return", "sharpe_ratio"],
            n_trials=8,
            storage_url=f"sqlite:///{temp_db_path}",
            direction=["maximize", "maximize"],
            n_jobs=1,
            seed=42,
        )

        # Create multi-objective objective
        objective = create_optimization_objective(
            backtest_engine=mock_backtest_engine,
            ticker="TEST",
            start_date="2020-01-01",
            end_date="2020-12-31",
            objective_type="multi",
            logger=mock_logger,
        )

        # Create optimization runner
        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=parameter_space,
            objective=objective,
            config=config,
            logger=mock_logger,
        )

        # Run optimization
        result = runner.optimize(n_trials=8, show_progress_bar=False)

        # Verify multi-objective results
        assert result is not None
        assert result.best_params is not None
        assert isinstance(result.best_value, float)

        # Verify parameters are still within bounds
        assert 1.0 <= result.best_params["leverage_base"] <= 3.0
        assert 5 <= result.best_params["ma_short"] <= 30

    def test_error_handling_and_robustness(
        self, mock_logger, mock_backtest_engine, parameter_space
    ) -> None:
        """Test error handling and robustness of the optimization system."""
        # Test with invalid configuration
        with pytest.raises(ValueError):
            _config = OptimizationConfig(
                study_name="",
                optimization_type="invalid_type",
                objective="total_return",
                n_trials=5,
                storage_url="sqlite:///:memory:",
            )

    def test_run_final_validation(self, mock_logger, mock_backtest_engine, parameter_space) -> None:
        """Test final validation with best parameters."""
        config = OptimizationConfig(
            study_name=f"validation_test_{int(datetime.now().timestamp())}",
            optimization_type="single_objective",
            objective="total_return",
            n_trials=5,
            storage_url="sqlite:///:memory:",
            direction="maximize",
        )

        objective = create_optimization_objective(
            backtest_engine=mock_backtest_engine,
            ticker="TEST",
            start_date="2020-01-01",
            end_date="2020-12-31",
            objective_type="single",
            logger=mock_logger,
        )

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=parameter_space,
            objective=objective,
            config=config,
            logger=mock_logger,
        )

        # Run a quick optimization first
        _result = runner.optimize(n_trials=3, show_progress_bar=False)

        # Run validation (should work even with minimal results)
        validation = runner.run_final_validation(n_validation_trials=2)

        assert validation is not None
        assert "n_validation_trials" in validation

    def test_performance_and_scalability(
        self, mock_logger, mock_backtest_engine, parameter_space
    ) -> None:
        """Test performance characteristics of the optimization system."""
        import time

        # Test with larger number of trials
        config = OptimizationConfig(
            study_name=f"perf_test_{int(datetime.now().timestamp())}",
            optimization_type="single_objective",
            objective="total_return",
            n_trials=20,  # More trials for performance testing
            storage_url="sqlite:///:memory:",
            direction="maximize",
            n_jobs=1,
        )

        objective = create_optimization_objective(
            backtest_engine=mock_backtest_engine,
            ticker="TEST",
            start_date="2020-01-01",
            end_date="2020-12-31",
            objective_type="single",
            logger=mock_logger,
        )

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=parameter_space,
            objective=objective,
            config=config,
            logger=mock_logger,
        )

        start_time = time.time()

        result = runner.optimize(n_trials=20, show_progress_bar=False)

        end_time = time.time()
        duration = end_time - start_time

        # Verify optimization completed
        assert result.n_trials == 20
        assert duration < 60  # Should complete within 60 seconds for testing

        # Performance assertions
        avg_time_per_trial = duration / 20
        assert avg_time_per_trial < 3  # Average less than 3 seconds per trial
