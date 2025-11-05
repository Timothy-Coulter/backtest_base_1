"""Tests for optimization runner functionality."""

import logging
import time
from unittest.mock import Mock, patch

import pytest

from backtester.optmisation.runner import OptimizationResult, OptimizationRunner


class TestOptimizationRunner:
    """Test cases for OptimizationRunner class."""

    def test_init_optimization_runner(self) -> None:
        """Test initializing OptimizationRunner."""
        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        assert runner.backtest_engine == mock_backtest_engine
        assert runner.parameter_space == mock_parameter_space
        assert runner.objective == mock_objective
        assert runner.config == mock_config
        assert runner.logger == logger

    def test_optimize_basic_functionality(self) -> None:
        """Test basic optimization functionality."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()
        mock_config.n_trials = 10
        mock_config.timeout = None
        mock_config.n_jobs = 1
        mock_config.study_name = "test_study"
        mock_config.get_storage_url.return_value = "sqlite:///:memory:"
        mock_config.direction = "maximize"

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Mock the study manager and optimization process
        mock_study = Mock()
        mock_study.best_params = {"param1": 1.5}
        mock_study.best_value = 0.15
        mock_study.trials = [Mock(), Mock()]  # 2 trials
        mock_study.direction = "maximize"
        mock_study.study_name = "test_study"

        with (
            patch.object(runner.study_manager, 'create_study', return_value=mock_study),
            patch.object(
                runner,
                '_run_standard_optimization',
                return_value=OptimizationResult(
                    best_params={"param1": 1.5},
                    best_value=0.15,
                    best_trial=None,
                    n_trials=10,
                    optimization_time=30.0,
                    study_summary={"study_name": "test_study"},
                    trial_statistics={"total_trials": 10, "complete_trials": 8},
                ),
            ) as mock_run_standard,
        ):
            result = runner.optimize(n_trials=10)

            assert isinstance(result, OptimizationResult)
            assert result.best_params["param1"] == 1.5
            assert result.best_value == 0.15
            mock_run_standard.assert_called_once()

    def test_optimize_with_timeout(self) -> None:
        """Test optimization with timeout."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()
        mock_config.n_trials = 50
        mock_config.timeout = 60
        mock_config.n_jobs = 1

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        mock_study = Mock()
        mock_study.best_params = {}
        mock_study.best_value = 0.0
        mock_study.trials = []
        mock_study.direction = "maximize"
        mock_study.study_name = "test_study"

        with (
            patch.object(runner.study_manager, 'create_study', return_value=mock_study),
            patch.object(
                runner,
                '_run_standard_optimization',
                return_value=OptimizationResult(
                    best_params={},
                    best_value=0.0,
                    best_trial=None,
                    n_trials=0,
                    optimization_time=65.0,
                    study_summary={},
                    trial_statistics={},
                ),
            ) as mock_run_standard,
        ):
            runner.optimize(timeout=60)

            mock_run_standard.assert_called_once()
            # Verify timeout was passed through
            call_args = mock_run_standard.call_args
            assert call_args[1]['timeout'] == 60

    def test_create_optuna_objective(self) -> None:
        """Test creation of Optuna objective function."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()
        mock_config.direction = "maximize"

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Mock backtest result
        mock_result = Mock()
        mock_result.value = 0.15
        mock_objective._run_backtest.return_value = mock_result
        mock_parameter_space.suggest_params.return_value = {"param1": 1.5}
        mock_objective.get_objective_functions.return_value = [Mock(return_value=0.15)]

        # Create the Optuna objective
        optuna_objective = runner._create_optuna_objective()

        # Mock Optuna trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.params = {"param1": 1.5}

        # Test successful objective evaluation
        result = optuna_objective(mock_trial)

        assert result == 0.15
        mock_parameter_space.suggest_params.assert_called_once_with(mock_trial)
        mock_objective._run_backtest.assert_called_once_with({"param1": 1.5})

    def test_create_optimization_result(self) -> None:
        """Test creation of optimization result."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Mock study and statistics
        mock_study = Mock()
        mock_study.best_params = {"param1": 1.5, "param2": 3}
        mock_study.best_value = 0.15
        mock_study.best_trial = Mock()
        mock_study.study_name = "test_study"
        mock_study.direction = "maximize"

        mock_stats = {
            "total_trials": 10,
            "complete_trials": 8,
            "pruned_trials": 1,
            "failed_trials": 1,
        }
        mock_summary = {"study_name": "test_study", "direction": "maximize"}

        runner.study_manager._study = mock_study
        runner.study_manager.get_trial_statistics.return_value = mock_stats
        runner.study_manager.get_study_summary.return_value = mock_summary

        runner.start_time = time.time() - 30  # 30 seconds ago

        result = runner._create_optimization_result()

        assert isinstance(result, OptimizationResult)
        assert result.best_params == {"param1": 1.5, "param2": 3}
        assert result.best_value == 0.15
        assert result.n_trials == 10
        assert result.optimization_time == 30.0
        assert result.study_summary == mock_summary
        assert result.trial_statistics == mock_stats

    def test_get_best_params(self) -> None:
        """Test getting best parameters."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Test with stored optimization result
        mock_result = OptimizationResult(
            best_params={"param1": 1.5},
            best_value=0.15,
            best_trial=None,
            n_trials=10,
            optimization_time=30.0,
            study_summary={},
            trial_statistics={},
        )
        runner.optimization_result = mock_result

        best_params = runner.get_best_params()
        assert best_params == {"param1": 1.5}

    def test_get_best_value(self) -> None:
        """Test getting best value."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Test with stored optimization result
        mock_result = OptimizationResult(
            best_params={"param1": 1.5},
            best_value=0.15,
            best_trial=None,
            n_trials=10,
            optimization_time=30.0,
            study_summary={},
            trial_statistics={},
        )
        runner.optimization_result = mock_result

        best_value = runner.get_best_value()
        assert best_value == 0.15

    def test_get_study_name(self) -> None:
        """Test getting study name."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()
        mock_config.study_name = "test_study"

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        study_name = runner.get_study_name()
        assert study_name == "test_study"

    def test_error_handling_in_optimization(self) -> None:
        """Test error handling during optimization."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()
        mock_config.n_trials = 5

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Mock study manager to raise an exception
        with (
            patch.object(
                runner.study_manager,
                'create_study',
                side_effect=RuntimeError("Study creation failed"),
            ),
            pytest.raises(RuntimeError, match="Study creation failed"),
        ):
            runner.optimize()

    def test_validate_params(self) -> None:
        """Test parameter validation."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock dependencies
        mock_backtest_engine = Mock()
        mock_parameter_space = Mock()
        mock_objective = Mock()
        mock_config = Mock()

        runner = OptimizationRunner(
            backtest_engine=mock_backtest_engine,
            parameter_space=mock_parameter_space,
            objective=mock_objective,
            config=mock_config,
            logger=logger,
        )

        # Test valid parameters
        valid_params = {"param1": 1.5, "param2": 3}
        assert runner.validate_params(valid_params) is True

        # Test invalid parameters (not a dictionary)
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            runner.validate_params("not_a_dict")

        # Test invalid parameter key
        with pytest.raises(ValueError, match="Parameter key must be string"):
            runner.validate_params({123: "value"})

        # Test invalid parameter value type
        with pytest.raises(ValueError, match="Parameter value must be one of"):
            runner.validate_params({"param": [1, 2, 3]})
