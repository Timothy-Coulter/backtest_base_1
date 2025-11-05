"""Tests for study management functionality."""

from unittest.mock import Mock, patch

from backtester.optmisation.parameter_space import (
    OptimizationConfig,
)
from backtester.optmisation.study_manager import (
    OptunaStudyManager,
)


class TestOptunaStudyManager:
    """Test cases for OptunaStudyManager class."""

    def test_init_study_manager(self) -> None:
        """Test initializing OptunaStudyManager."""
        import logging

        logger = logging.getLogger(__name__)
        manager = OptunaStudyManager(logger=logger)
        assert manager.logger == logger

    def test_create_study_single_objective(self) -> None:
        """Test creating a single objective study."""
        import logging

        logger = logging.getLogger(__name__)
        manager = OptunaStudyManager(
            study_name="test_single_objective",
            storage_url="sqlite:///:memory:",
            direction="maximize",
            logger=logger,
        )

        config = OptimizationConfig()

        with patch("optuna.create_study") as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study

            study = manager.create_study(config)

            assert study == mock_study
            mock_create_study.assert_called_once()

    def test_load_existing_study(self) -> None:
        """Test loading an existing study."""
        import logging

        logger = logging.getLogger(__name__)
        manager = OptunaStudyManager(
            study_name="existing_study",
            storage_url="sqlite:///:memory:",
            direction="maximize",
            logger=logger,
        )

        config = OptimizationConfig()

        with patch("optuna.load_study") as mock_load_study:
            mock_study = Mock()
            mock_load_study.return_value = mock_study

            study = manager.load_study(config)

            assert study == mock_study
            mock_load_study.assert_called_once()

    def test_get_study_summary(self) -> None:
        """Test getting study summary."""
        import logging

        logger = logging.getLogger(__name__)
        manager = OptunaStudyManager(logger=logger)

        # Create a mock study with some trials
        mock_study = Mock()
        mock_trial_1 = Mock()
        mock_trial_1.params = {"param1": 1.0, "param2": 2}
        mock_trial_1.value = 0.15
        mock_trial_1.state = "COMPLETE"

        mock_trial_2 = Mock()
        mock_trial_2.params = {"param1": 1.5, "param2": 3}
        mock_trial_2.value = 0.20
        mock_trial_2.state = "COMPLETE"

        mock_study.trials = [mock_trial_1, mock_trial_2]
        mock_study.best_params = {"param1": 1.5, "param2": 3}
        mock_study.best_value = 0.20
        mock_study.study_name = "test_study"
        mock_study.direction = "maximize"

        manager._study = mock_study

        summary = manager.get_study_summary()

        assert summary["study_name"] == "test_study"
        assert summary["n_trials"] == 2
        assert summary["best_params"] == {"param1": 1.5, "param2": 3}
        assert summary["best_value"] == 0.20
