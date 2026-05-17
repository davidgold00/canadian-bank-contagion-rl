from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .allocation_policy import (
    BANKS,
    CASH_ASSET,
    AllocationPolicyResult,
    available_assets,
    generate_model_allocation,
    load_ppo_model,
)


@dataclass(frozen=True)
class SimulationResult:
    """Full paper-portfolio output used by the dashboard and tests."""

    ledger: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    weights: pd.DataFrame
    benchmarks: pd.DataFrame
    current_holdings: pd.DataFrame
    policy_source: str


class PaperPortfolioSimulator:
    """Daily long-only paper portfolio simulator with an auditable trade ledger.

    The simulator trades at the current day's observed close using only information
    available up to that date. Returns from day t to day t+1 are earned by the
    holdings established on day t.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        assets: Sequence[str] | None = None,
        model_path: str | Path | None = None,
        use_trained_model: bool = True,
        long_only: bool = True,
    ) -> None:
        self.prices = self._prepare_prices(prices)
        self.features = self._prepare_features(features, self.prices.index)
        self.assets = available_assets(self.prices, assets)
        self.risky_assets = [asset for asset in self.assets if asset != CASH_ASSET]
        self.model_path = model_path
        self.use_trained_model = use_trained_model
        self.long_only = long_only

    @staticmethod
    def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
        out = prices.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.set_index("date")
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        numeric = out.apply(pd.to_numeric, errors="coerce")
        return numeric.ffill().bfill().dropna(how="all")

    @staticmethod
    def _prepare_features(features: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        out = features.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.set_index("date")
        out.index = pd.to_datetime(out.index)
        out = out.sort_index().apply(pd.to_numeric, errors="coerce")
        return out.reindex(index).ffill().bfill()

    def simulate(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
        rebalance_threshold: float = 0.01,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        max_single_name_weight: float = 0.22,
        max_bank_exposure: float = 0.80,
        defensive_cash_sensitivity: float = 1.0,
    ) -> SimulationResult:
        """Run the daily paper-fund simulation."""
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")

        dates = self._simulation_dates(start_date, end_date)
        if len(dates) == 0:
            raise ValueError("No simulation dates are available after filtering.")

        ppo_model = None
        ppo_note = "PPO not requested."
        if self.use_trained_model:
            ppo_model, ppo_note = load_ppo_model(self.model_path)

        transaction_cost_rate = transaction_cost_bps / 10_000
        shares = pd.Series(0.0, index=self.risky_assets, dtype=float)
        avg_cost = pd.Series(0.0, index=self.risky_assets, dtype=float)
        cash = float(initial_capital)
        previous_value = float(initial_capital)
        first_rebalance = True
        policy_sources: list[str] = []

        ledger_rows: list[dict] = []
        trade_rows: list[dict] = []
        holding_rows: list[dict] = []
        weight_rows: list[pd.Series] = []

        for date in dates:
            price_today = self.prices.reindex(columns=self.risky_assets).loc[date].astype(float)
            pre_trade_asset_values = shares * price_today
            pre_trade_value = float(cash + pre_trade_asset_values.sum())
            pre_trade_value = max(pre_trade_value, 0.0)
            daily_pnl_before_cost = pre_trade_value - previous_value

            current_weights = self._current_weights(cash, pre_trade_asset_values, pre_trade_value)
            allocation = generate_model_allocation(
                prices_history=self.prices.loc[:date],
                features_history=self.features.loc[:date],
                assets=self.assets,
                current_weights=current_weights,
                max_single_name_weight=max_single_name_weight,
                max_bank_exposure=max_bank_exposure,
                defensive_cash_sensitivity=defensive_cash_sensitivity,
                use_trained_model=self.use_trained_model,
                ppo_model=ppo_model,
            )
            target_weights = allocation.weights.reindex(self.assets).fillna(0.0)
            target_values = self._target_values_with_cost_reserve(
                target_weights,
                pre_trade_asset_values,
                pre_trade_value,
                transaction_cost_rate,
            )

            trade_values = target_values.reindex(self.risky_assets).fillna(0.0) - pre_trade_asset_values
            if not first_rebalance:
                small = (trade_values.abs() / max(pre_trade_value, 1.0)) < rebalance_threshold
                trade_values.loc[small] = 0.0
            first_rebalance = False

            day_costs = 0.0
            day_turnover_notional = 0.0
            day_trade_count = 0
            for asset, trade_value in trade_values.items():
                price = float(price_today[asset])
                if abs(trade_value) < 1e-8 or price <= 0 or np.isnan(price):
                    continue
                action = "BUY" if trade_value > 0 else "SELL"
                share_delta = trade_value / price
                transaction_cost = abs(trade_value) * transaction_cost_rate
                old_shares = float(shares[asset])

                if self.long_only and old_shares + share_delta < -1e-8:
                    share_delta = -old_shares
                    trade_value = share_delta * price
                    transaction_cost = abs(trade_value) * transaction_cost_rate
                    action = "SELL"

                cash -= trade_value + transaction_cost
                shares[asset] += share_delta
                day_costs += transaction_cost
                day_turnover_notional += abs(trade_value)
                day_trade_count += 1

                if action == "BUY" and shares[asset] > 0:
                    existing_cost = avg_cost[asset] * max(old_shares, 0.0)
                    avg_cost[asset] = (existing_cost + abs(trade_value) + transaction_cost) / shares[asset]
                elif shares[asset] <= 1e-10:
                    shares[asset] = 0.0
                    avg_cost[asset] = 0.0

                trade_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "action": action,
                        "shares": float(abs(share_delta)),
                        "price": price,
                        "notional": float(abs(trade_value)),
                        "transaction_cost": float(transaction_cost),
                        "reason": allocation.reasons.get(asset, "Rebalanced to target weights."),
                        "target_weight": float(target_weights.get(asset, 0.0)),
                        "pre_trade_weight": float(current_weights.get(asset, 0.0)),
                    }
                )

            if self.long_only and cash < -1e-6:
                repair_cost, repair_trades = self._raise_cash(date, shares, avg_cost, price_today, -cash, transaction_cost_rate)
                cash = 0.0
                day_costs += repair_cost
                day_turnover_notional += sum(row["notional"] for row in repair_trades)
                day_trade_count += len(repair_trades)
                trade_rows.extend(repair_trades)

            post_values = shares * price_today
            portfolio_value = float(cash + post_values.sum())
            actual_weights = self._current_weights(cash, post_values, portfolio_value)
            bank_exposure = float(actual_weights.reindex([b for b in BANKS if b in actual_weights.index]).fillna(0.0).sum())
            turnover = day_turnover_notional / max(pre_trade_value, 1.0)
            daily_pnl = portfolio_value - previous_value
            daily_return = daily_pnl / previous_value if previous_value else 0.0
            risk_score = float(allocation.diagnostics.get("contagion_risk_score", 50.0))

            ledger_rows.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "cash": float(cash),
                    "daily_return": float(daily_return),
                    "daily_pnl": float(daily_pnl),
                    "turnover": float(turnover),
                    "transaction_costs": float(day_costs),
                    "contagion_risk_score": risk_score,
                    "cash_weight": float(actual_weights.get(CASH_ASSET, 0.0)),
                    "bank_exposure": bank_exposure,
                    "number_of_trades": day_trade_count,
                    "policy_source": allocation.policy_source,
                    "policy_note": ppo_note if allocation.policy_source != "trained PPO model" else "PPO inference succeeded.",
                    "allocation_observation_date": allocation.diagnostics.get("observation_date", date),
                }
            )
            policy_sources.append(allocation.policy_source)
            weight_rows.append(actual_weights.rename(date))

            for asset in self.risky_assets:
                market_value = float(post_values.get(asset, 0.0))
                holding_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "shares": float(shares[asset]),
                        "latest_price": float(price_today[asset]),
                        "market_value": market_value,
                        "weight": float(actual_weights.get(asset, 0.0)),
                        "average_cost": float(avg_cost[asset]),
                        "unrealized_pnl": float((price_today[asset] - avg_cost[asset]) * shares[asset]) if shares[asset] else 0.0,
                    }
                )
            holding_rows.append(
                {
                    "date": date,
                    "asset": CASH_ASSET,
                    "shares": 0.0,
                    "latest_price": 1.0,
                    "market_value": float(cash),
                    "weight": float(actual_weights.get(CASH_ASSET, 0.0)),
                    "average_cost": 1.0,
                    "unrealized_pnl": 0.0,
                }
            )

            previous_value = portfolio_value

        ledger = pd.DataFrame(ledger_rows).set_index("date")
        trades = pd.DataFrame(trade_rows)
        if not trades.empty:
            trades["date"] = pd.to_datetime(trades["date"])
        else:
            trades = pd.DataFrame(
                columns=[
                    "date",
                    "asset",
                    "action",
                    "shares",
                    "price",
                    "notional",
                    "transaction_cost",
                    "reason",
                    "target_weight",
                    "pre_trade_weight",
                ]
            )
        holdings = pd.DataFrame(holding_rows)
        holdings["date"] = pd.to_datetime(holdings["date"])
        weights = pd.DataFrame(weight_rows).fillna(0.0)
        weights.index.name = "date"
        benchmarks = self._build_benchmarks(dates, initial_capital)
        current_holdings = holdings.loc[holdings["date"] == holdings["date"].max()].copy()
        policy_source = "trained PPO model" if "trained PPO model" in policy_sources else "rule-based fallback policy"

        return SimulationResult(
            ledger=ledger,
            trades=trades,
            holdings=holdings,
            weights=weights,
            benchmarks=benchmarks,
            current_holdings=current_holdings,
            policy_source=policy_source,
        )

    def _simulation_dates(self, start_date: str | pd.Timestamp | None, end_date: str | pd.Timestamp | None) -> pd.DatetimeIndex:
        index = self.prices.index
        if start_date is not None:
            index = index[index >= pd.Timestamp(start_date)]
        if end_date is not None:
            index = index[index <= pd.Timestamp(end_date)]
        return pd.DatetimeIndex(index)

    def _current_weights(self, cash: float, asset_values: pd.Series, total_value: float) -> pd.Series:
        weights = pd.Series(0.0, index=self.assets, dtype=float)
        if total_value <= 0:
            weights.loc[CASH_ASSET] = 1.0
            return weights
        for asset, value in asset_values.items():
            weights.loc[asset] = float(value / total_value)
        weights.loc[CASH_ASSET] = float(cash / total_value)
        return weights.fillna(0.0)

    def _target_values_with_cost_reserve(
        self,
        target_weights: pd.Series,
        current_asset_values: pd.Series,
        portfolio_value: float,
        transaction_cost_rate: float,
    ) -> pd.Series:
        target_values = target_weights.reindex(self.assets).fillna(0.0) * portfolio_value
        trade_values = target_values.reindex(self.risky_assets).fillna(0.0) - current_asset_values
        estimated_cost = float(trade_values.abs().sum() * transaction_cost_rate)
        target_cash = float(target_values.get(CASH_ASSET, 0.0))

        if estimated_cost > target_cash and portfolio_value > estimated_cost:
            non_cash_budget = max(portfolio_value - estimated_cost, 0.0)
            non_cash_target = target_values.reindex(self.risky_assets).fillna(0.0)
            if non_cash_target.sum() > 0:
                target_values.loc[self.risky_assets] = non_cash_target / non_cash_target.sum() * non_cash_budget
            target_values.loc[CASH_ASSET] = estimated_cost
        return target_values

    def _raise_cash(
        self,
        date: pd.Timestamp,
        shares: pd.Series,
        avg_cost: pd.Series,
        prices: pd.Series,
        cash_needed: float,
        transaction_cost_rate: float,
    ) -> tuple[float, list[dict]]:
        rows: list[dict] = []
        total_cost = 0.0
        remaining = cash_needed
        market_values = (shares * prices).sort_values(ascending=False)
        for asset, value in market_values.items():
            if remaining <= 1e-8 or value <= 0:
                break
            gross_sale = min(float(value), remaining / max(1 - transaction_cost_rate, 1e-8))
            share_delta = gross_sale / float(prices[asset])
            shares[asset] = max(shares[asset] - share_delta, 0.0)
            if shares[asset] <= 1e-10:
                shares[asset] = 0.0
                avg_cost[asset] = 0.0
            cost = gross_sale * transaction_cost_rate
            proceeds = gross_sale - cost
            remaining -= proceeds
            total_cost += cost
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "action": "SELL",
                    "shares": float(share_delta),
                    "price": float(prices[asset]),
                    "notional": float(gross_sale),
                    "transaction_cost": float(cost),
                    "reason": "Sold a small amount to prevent negative cash after transaction costs.",
                    "target_weight": 0.0,
                    "pre_trade_weight": 0.0,
                }
            )
        return total_cost, rows

    def _build_benchmarks(self, dates: pd.DatetimeIndex, initial_capital: float) -> pd.DataFrame:
        prices = self.prices.reindex(dates).ffill().bfill()
        returns = prices[self.risky_assets].pct_change().fillna(0.0)
        out = pd.DataFrame(index=dates)

        bank_cols = [bank for bank in BANKS if bank in returns.columns]
        if bank_cols:
            out["Equal-weight Big Six"] = initial_capital * (1 + returns[bank_cols].mean(axis=1)).cumprod()

        for asset, label in [("XFN.TO", "XFN buy-and-hold"), ("XIU.TO", "XIU buy-and-hold"), ("XIC.TO", "XIC buy-and-hold")]:
            if asset in prices.columns:
                series = prices[asset] / prices[asset].iloc[0] * initial_capital
                out[label] = series

        out["Cash"] = initial_capital
        return out
