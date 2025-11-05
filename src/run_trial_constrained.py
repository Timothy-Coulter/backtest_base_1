# ====================================
# MULTI-OBJECTIVE OPTIMIZATION
# ====================================

"""Constrained portfolio trial optimization.

This module provides functionality for running constrained portfolio
optimization trials using Optuna with multi-objective optimization.
"""

import optuna

from run_sim import run_portfolio_simulation  # type: ignore[import-not-found]
from utils import get_data  # type: ignore[import-not-found]

if __name__ == '__main__':
    MAX_DRAWDOWN_LIMIT = -0.5  # constraint: drawdown must not exceed -30%

    ticker = "SPY"
    start_date = "1990-01-01"
    end_date = "2025-11-01"
    interval = "1mo"

    data = get_data(ticker=ticker, start_date=start_date, end_date=end_date, interval=interval)

    def objective(trial):
        """Objective function for constrained optimization."""
        result = run_portfolio_simulation(
            data=data,
            leverage_base=trial.suggest_float("leverage_base", 1, 2),
            leverage_alpha=trial.suggest_float("leverage_alpha", 1.0, 10.0),
            base_to_alpha_split=trial.suggest_float("base_to_alpha_split", 0.01, 0.99),
            alpha_to_base_split=trial.suggest_float("alpha_to_base_split", 0.01, 0.99),
            stop_loss_base=trial.suggest_float("stop_loss_base", 0.01, 0.1),
            stop_loss_alpha=trial.suggest_float("stop_loss_alpha", 0.01, 0.1),
            take_profit_target=trial.suggest_float("take_profit_target", 0.01, 0.99),
        )

        # We want to maximize total_return and minimize drawdown (more positive = better)
        total_return = result["total_return"]
        drawdown = result["max_drawdown"]
        print(drawdown)
        # constraint: penalize if drawdown < limit
        constraint_violation = 9999 if abs(drawdown) > abs(MAX_DRAWDOWN_LIMIT) else -1
        trial.set_user_attr("result", result)
        return total_return, abs(drawdown), constraint_violation

    study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print("Best trade-offs:")
    for trial in study.best_trials:
        print("Parameters:", trial.params)
        print("Values:", trial.values)
        result = trial.user_attrs["result"]
        print("Total Return: {:.2%}".format(result["total_return"]))
        print("Max Drawdown: {:.2%}".format(result["max_drawdown"]))
        print("Sharpe Ratio: {:.3f}".format(result.get("sharpe_ratio", 0)))
        print("Calmar Ratio: {:.3f}".format(result.get("calmar_ratio", 0)))
        print("Volatility: {:.2%}".format(result["volatility"]))
        print("Number of Trades:", result["total_trades"])
        print("Win Rate: {:.2%}".format(result.get("win_rate", 0)))
        print("Profit Factor:", result.get("profit_factor", 0))
        print("=" * 30)
