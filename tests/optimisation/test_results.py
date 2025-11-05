"""Tests for results analysis functionality."""

from unittest.mock import Mock

import pytest

from backtester.optmisation.results import ResultsAnalyzer
from backtester.optmisation.runner import OptimizationResult


class TestResultsAnalyzer:
    """Test cases for ResultsAnalyzer class."""

    def test_init_results_analyzer(self) -> None:
        """Test initializing ResultsAnalyzer."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)
        assert analyzer.logger == logger

    def test_init_results_analyzer_default_logger(self) -> None:
        """Test initializing ResultsAnalyzer with default logger."""
        analyzer = ResultsAnalyzer()
        assert analyzer.logger is not None

    def test_analyze_optimization_result(self) -> None:
        """Test analyzing optimization result."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        # Create mock optimization result
        mock_result = OptimizationResult(
            best_params={"leverage_base": 2.0, "ma_short": 10},
            best_value=0.15,
            best_trial=None,
            n_trials=10,
            optimization_time=30.0,
            study_summary={"study_name": "test_study", "direction": "maximize"},
            trial_statistics={
                "total_trials": 10,
                "complete_trials": 8,
                "pruned_trials": 1,
                "failed_trials": 1,
                "completion_rate": 0.8,
                "min_value": 0.10,
                "max_value": 0.20,
                "mean_value": 0.15,
                "value_std": 0.025,
            },
        )

        # Create mock backtest engine
        mock_backtest_engine = Mock()
        mock_backtest_results = {
            "performance": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.05,
                "volatility": 0.12,
                "calmar_ratio": 3.0,
                "total_trades": 100,
                "win_rate": 0.55,
                "profit_factor": 1.5,
            }
        }
        mock_backtest_engine.run_backtest.return_value = mock_backtest_results

        analysis = analyzer.analyze_optimization_result(
            result=mock_result,
            backtest_engine=mock_backtest_engine,
            ticker="SPY",
            start_date="2020-01-01",
            end_date="2020-12-31",
            interval="1mo",
        )

        # Verify analysis structure
        assert "optimization_summary" in analysis
        assert "parameter_analysis" in analysis
        assert "performance_analysis" in analysis
        assert "trial_analysis" in analysis
        assert "validation_backtest" in analysis
        assert "recommendations" in analysis

    def test_get_optimization_summary(self) -> None:
        """Test getting optimization summary."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={"param1": 1.5},
            best_value=0.15,
            best_trial=None,
            n_trials=10,
            optimization_time=30.0,
            study_summary={"study_name": "test_study", "direction": "maximize"},
            trial_statistics={
                "complete_trials": 8,
                "completion_rate": 0.8,
            },
        )

        summary = analyzer._get_optimization_summary(mock_result)

        assert summary["study_name"] == "test_study"
        assert summary["optimization_direction"] == "maximize"
        assert summary["total_trials"] == 10
        assert summary["complete_trials"] == 8
        assert summary["completion_rate"] == 0.8
        assert summary["optimization_time_seconds"] == 30.0
        assert summary["best_value"] == 0.15
        assert summary["best_parameters"] == {"param1": 1.5}

    def test_analyze_parameters(self) -> None:
        """Test parameter analysis."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={
                "leverage_base": 2.0,
                "ma_short": 10,
                "ma_long": 50,
                "initial_capital": 10000,
                "commission_rate": 0.001,
            },
            best_value=0.15,
            best_trial=None,
            n_trials=5,
            optimization_time=20.0,
            study_summary={},
            trial_statistics={},
        )

        param_analysis = analyzer._analyze_parameters(mock_result)

        assert "best_parameters" in param_analysis
        assert "strategy_parameters" in param_analysis
        assert "portfolio_parameters" in param_analysis
        assert "other_parameters" in param_analysis
        assert param_analysis["parameter_count"] == 5

        # Check parameter categorization
        assert "leverage_base" in param_analysis["strategy_parameters"]
        assert "ma_short" in param_analysis["strategy_parameters"]
        assert "initial_capital" in param_analysis["portfolio_parameters"]

    def test_analyze_parameters_no_params(self) -> None:
        """Test parameter analysis with no parameters."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=0,
            optimization_time=0.0,
            study_summary={},
            trial_statistics={},
        )

        param_analysis = analyzer._analyze_parameters(mock_result)

        assert param_analysis["error"] == "No parameters available for analysis"

    def test_analyze_performance_metrics(self) -> None:
        """Test performance metrics analysis."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_trial = Mock()
        mock_trial.user_attrs = {"sharpe_ratio": 1.2, "max_drawdown": -0.05}

        mock_result = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=mock_trial,
            n_trials=10,
            optimization_time=30.0,
            study_summary={},
            trial_statistics={
                "min_value": 0.10,
                "max_value": 0.20,
                "mean_value": 0.15,
                "value_std": 0.025,
            },
        )

        perf_analysis = analyzer._analyze_performance_metrics(mock_result)

        assert "best_trial_metrics" in perf_analysis
        assert "metric_statistics" in perf_analysis
        assert perf_analysis["best_trial_metrics"]["sharpe_ratio"] == 1.2
        assert perf_analysis["best_trial_metrics"]["max_drawdown"] == -0.05

    def test_analyze_performance_metrics_no_trial(self) -> None:
        """Test performance analysis with no best trial."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=5,
            optimization_time=20.0,
            study_summary={},
            trial_statistics={},
        )

        perf_analysis = analyzer._analyze_performance_metrics(mock_result)

        assert perf_analysis["error"] == "No best trial available for analysis"

    def test_analyze_trial_results(self) -> None:
        """Test trial results analysis."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=10,
            optimization_time=30.0,
            study_summary={},
            trial_statistics={
                "total_trials": 10,
                "complete_trials": 8,
                "pruned_trials": 1,
                "failed_trials": 1,
                "completion_rate": 0.8,
            },
        )

        trial_analysis = analyzer._analyze_trial_results(mock_result)

        assert "trial_outcomes" in trial_analysis
        assert "trial_quality" in trial_analysis

        outcomes = trial_analysis["trial_outcomes"]
        assert outcomes["total_trials"] == 10
        assert outcomes["successful_trials"] == 8
        assert outcomes["pruned_trials"] == 1
        assert outcomes["failed_trials"] == 1

        quality = trial_analysis["trial_quality"]
        assert quality["success_rate"] == 0.8

    def test_run_validation_backtest_success(self) -> None:
        """Test successful validation backtest."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_backtest_engine = Mock()
        mock_backtest_results = {
            "performance": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.05,
                "volatility": 0.12,
                "calmar_ratio": 3.0,
                "total_trades": 100,
                "win_rate": 0.55,
                "profit_factor": 1.5,
            }
        }
        mock_backtest_engine.run_backtest.return_value = mock_backtest_results

        mock_result = OptimizationResult(
            best_params={"leverage_base": 2.0},
            best_value=0.15,
            best_trial=None,
            n_trials=5,
            optimization_time=20.0,
            study_summary={},
            trial_statistics={},
        )

        validation = analyzer._run_validation_backtest(
            result=mock_result,
            backtest_engine=mock_backtest_engine,
            ticker="SPY",
            start_date="2020-01-01",
            end_date="2020-12-31",
            interval="1mo",
        )

        assert validation["backtest_successful"] is True
        assert "performance_metrics" in validation
        assert "summary" in validation
        assert validation["summary"]["total_return"] == 0.15
        assert validation["summary"]["sharpe_ratio"] == 1.2

    def test_run_validation_backtest_failure(self) -> None:
        """Test failed validation backtest."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_backtest_engine = Mock()
        mock_backtest_engine.run_backtest.side_effect = RuntimeError("Backtest failed")

        mock_result = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=5,
            optimization_time=20.0,
            study_summary={},
            trial_statistics={},
        )

        validation = analyzer._run_validation_backtest(
            result=mock_result,
            backtest_engine=mock_backtest_engine,
            ticker="SPY",
            start_date="2020-01-01",
            end_date="2020-12-31",
            interval="1mo",
        )

        assert validation["backtest_successful"] is False
        assert "error" in validation
        assert "Backtest failed" in validation["error"]

    def test_generate_recommendations(self) -> None:
        """Test generating recommendations."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={"param1": 1.0},
            best_value=-0.05,  # Negative value for maximize direction
            best_trial=None,
            n_trials=10,
            optimization_time=400.0,  # Long optimization time
            study_summary={"direction": "maximize"},
            trial_statistics={
                "completion_rate": 0.7,  # Below 0.8 threshold
                "failed_trials": 2,
            },
        )

        recommendations = analyzer._generate_recommendations(mock_result)

        assert "recommendations" in recommendations
        assert "optimization_grade" in recommendations
        assert "next_steps" in recommendations

        rec_list = recommendations["recommendations"]

        # Should have recommendations for low completion rate and negative best value
        completion_recs = [r for r in rec_list if r["type"] == "optimization_quality"]
        assert len(completion_recs) > 0

        value_recs = [r for r in rec_list if r["type"] == "objective_function"]
        assert len(value_recs) > 0

    def test_estimate_parameter_importance(self) -> None:
        """Test parameter importance estimation."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={
                "leverage_base": 3.0,  # Far from default (1.0)
                "leverage_alpha": 6.0,  # Far from default (3.0)
                "ma_short": 15,
                "some_param": 0.5,  # Default importance
            },
            best_value=0.15,
            best_trial=None,
            n_trials=5,
            optimization_time=20.0,
            study_summary={},
            trial_statistics={},
        )

        importance = analyzer._estimate_parameter_importance(mock_result)

        assert "leverage_base" in importance
        assert "leverage_alpha" in importance
        assert "some_param" in importance

        # leverage_base should have high importance (3.0 - 1.0 = 2.0, normalized)
        assert importance["leverage_base"] > importance["some_param"]
        assert importance["leverage_alpha"] > importance["some_param"]

    def test_assess_distribution_quality(self) -> None:
        """Test distribution quality assessment."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        # Test degenerate distribution
        stats_degenerate = {"mean_value": 0.0, "value_std": 0.0}
        quality = analyzer._assess_distribution_quality(stats_degenerate)
        assert quality == "degenerate"

        # Test poor dispersion
        stats_poor = {"mean_value": 0.005, "value_std": 0.001}
        quality = analyzer._assess_distribution_quality(stats_poor)
        assert quality == "poor_dispersion"

        # Test good dispersion
        stats_good = {"mean_value": 0.1, "value_std": 0.05}
        quality = analyzer._assess_distribution_quality(stats_good)
        assert quality == "good_dispersion"

    def test_grade_optimization(self) -> None:
        """Test optimization grading."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        # Test Grade A
        mock_result_a = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=50,
            optimization_time=60.0,
            study_summary={},
            trial_statistics={"completion_rate": 0.9},
        )
        grade = analyzer._grade_optimization(mock_result_a)
        assert grade == "A"

        # Test Grade B
        mock_result_b = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=30,
            optimization_time=60.0,
            study_summary={},
            trial_statistics={"completion_rate": 0.8},
        )
        grade = analyzer._grade_optimization(mock_result_b)
        assert grade == "B"

        # Test Grade F
        mock_result_f = OptimizationResult(
            best_params={},
            best_value=0.15,
            best_trial=None,
            n_trials=5,
            optimization_time=60.0,
            study_summary={},
            trial_statistics={"completion_rate": 0.4},
        )
        grade = analyzer._grade_optimization(mock_result_f)
        assert grade == "F"

    def test_export_analysis_report_markdown(self) -> None:
        """Test exporting analysis report as markdown."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        analysis = {
            "optimization_summary": {
                "study_name": "test_study",
                "optimization_direction": "maximize",
                "total_trials": 10,
                "completion_rate": 0.8,
                "best_value": 0.15,
                "optimization_time_seconds": 30.0,
            },
            "parameter_analysis": {"best_parameters": {"param1": 1.5}},
            "performance_analysis": {},
            "trial_analysis": {},
            "validation_backtest": {},
            "recommendations": {
                "recommendations": [{"priority": "high", "message": "Test recommendation"}]
            },
        }

        report = analyzer.export_analysis_report(analysis, format="markdown")

        assert "# Optimization Analysis Report" in report
        assert "test_study" in report
        assert "maximize" in report
        assert "param1: 1.5" in report
        assert "Test recommendation" in report

    def test_export_analysis_report_json(self) -> None:
        """Test exporting analysis report as JSON."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        analysis = {"key": "value", "number": 42}

        report = analyzer.export_analysis_report(analysis, format="json")

        # Should be valid JSON
        import json

        parsed = json.loads(report)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_export_analysis_report_unsupported_format(self) -> None:
        """Test exporting with unsupported format."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        analysis = {"key": "value"}

        with pytest.raises(ValueError, match="Unsupported export format"):
            analyzer.export_analysis_report(analysis, format="unsupported")

    def test_suggest_next_steps(self) -> None:
        """Test suggesting next steps."""
        import logging

        logger = logging.getLogger(__name__)
        analyzer = ResultsAnalyzer(logger=logger)

        mock_result = OptimizationResult(
            best_params={},
            best_value=-0.1,  # Negative for maximize
            best_trial=None,
            n_trials=5,
            optimization_time=400.0,  # Long time
            study_summary={"direction": "maximize"},
            trial_statistics={
                "completion_rate": 0.7,  # Below threshold
                "failed_trials": 1,
            },
        )

        suggestions = analyzer._suggest_next_steps(mock_result)

        # Should suggest improvements
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Should contain relevant suggestions
        suggestion_text = " ".join(suggestions)
        assert "constraint" in suggestion_text.lower() or "parameter" in suggestion_text.lower()
        assert "efficiency" in suggestion_text.lower() or "time" in suggestion_text.lower()
        assert "validation" in suggestion_text.lower()
