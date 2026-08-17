import unittest
from unittest.mock import patch
import pandas as pd
import pytest

from tradingagents.agents.utils.ma_crossover_tool import get_ma_crossover


@pytest.mark.unit
class TestMACrossoverTool(unittest.TestCase):

    def test_rejects_invalid_periods(self):
        """Verifica che venga restituito un errore se fast_period >= slow_period."""
        result = get_ma_crossover.invoke({
            "symbol": "AAPL",
            "curr_date": "2026-01-10",
            "fast_period": 50,
            "slow_period": 20,
        })
        self.assertIn("Error: fast_period (50) must be smaller than slow_period (20)", result)

    @patch("tradingagents.agents.utils.ma_crossover_tool._get_stock_stats_bulk")
    def test_computes_bullish_trend_and_golden_cross(self, mock_get_stats):
        """Simula dati storici e verifica il calcolo di un trend bullish e Golden Cross."""
        def mock_stats(symbol, indicator, curr_date):
            if "20_sma" in indicator:
                return {
                    "2026-01-01": 100.0,
                    "2026-01-02": 105.0,
                    "2026-01-03": 110.0,
                }
            return {
                "2026-01-01": 102.0,
                "2026-01-02": 103.0,
                "2026-01-03": 104.0,
            }

        mock_get_stats.side_effect = mock_stats

        result = get_ma_crossover.invoke({
            "symbol": "AAPL",
            "curr_date": "2026-01-03",
            "fast_period": 20,
            "slow_period": 50,
        })

        self.assertIn("Current trend: bullish", result)
        self.assertIn("Golden cross on 2026-01-02", result)
        self.assertIn("1 trading days ago", result)

    @patch("tradingagents.agents.utils.ma_crossover_tool._get_stock_stats_bulk")
    def test_handles_insufficient_history(self, mock_get_stats):
        """Verifica la gestione di dati storici insufficienti o valori N/A."""
        mock_get_stats.return_value = {
            "2026-01-01": "N/A",
            "2026-01-02": 100.0,
        }

        result = get_ma_crossover.invoke({
            "symbol": "AAPL",
            "curr_date": "2026-01-02",
            "fast_period": 20,
            "slow_period": 50,
        })

        self.assertIn("Not enough history to compute a 20/50 crossover", result)

if __name__ == "__main__":
    unittest.main()