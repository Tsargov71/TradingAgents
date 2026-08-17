import unittest
from unittest.mock import patch
import pandas as pd
import pytest

from tradingagents.agents.utils.ma_crossover_tool import get_ma_crossover
from tradingagents.agents.utils.volatility_tool import get_volatility_analysis


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


@pytest.mark.unit
class TestVolatilityTool(unittest.TestCase):

    @patch("tradingagents.agents.utils.volatility_tool._get_volatility_frame")
    def test_computes_volatility_analysis(self, mock_get_frame):
        """Verifica che la generazione del report di volatilitÃ  funzioni correttamente con dati validi."""
        # Crea uno storico mock di 25 righe (superiore al requisito minimo atr_lookback + 1 = 21)
        dates = pd.date_range(end="2026-01-04", periods=25).strftime("%Y-%m-%d").tolist()
        mock_get_frame.return_value = pd.DataFrame({
            "Date": dates,
            "Close": [150.0] * 25,
            "boll": [150.0] * 25,
            "boll_ub": [160.0] * 25,
            "boll_lb": [140.0] * 25,
            "atr": [2.5] * 25,
        })

        result = get_volatility_analysis.invoke({
            "symbol": "AAPL",
            "curr_date": "2026-01-04",
            "atr_lookback": 20,
            "atr_stop_multiple": 2.0,
        })

        self.assertIsInstance(result, str)
        self.assertIn("# Volatility Analysis for AAPL", result)
        self.assertIn("Close: 150.00", result)
        self.assertIn("Suggested stop-loss", result)
        self.assertNotIn("Error", result)

    @patch("tradingagents.agents.utils.volatility_tool._get_volatility_frame")
    def test_volatility_handles_data_error(self, mock_get_frame):
        """Verifica la gestione dell'eccezione se il caricamento del frame fallisce."""
        mock_get_frame.side_effect = Exception("Data loading error")

        result = get_volatility_analysis.invoke({
            "symbol": "AAPL",
            "curr_date": "2026-01-04",
        })

        self.assertIn("Error computing volatility analysis for AAPL", result)

    @patch("tradingagents.agents.utils.volatility_tool._get_volatility_frame")
    def test_volatility_handles_insufficient_data(self, mock_get_frame):
        """Verifica che avvisi l'utente se la lunghezza del dataframe Ã¨ inferiore a atr_lookback + 1."""
        dates = pd.date_range(end="2026-01-04", periods=5).strftime("%Y-%m-%d").tolist()
        mock_get_frame.return_value = pd.DataFrame({
            "Date": dates,
            "Close": [150.0] * 5,
            "boll": [150.0] * 5,
            "boll_ub": [160.0] * 5,
            "boll_lb": [140.0] * 5,
            "atr": [2.5] * 5,
        })

        result = get_volatility_analysis.invoke({
            "symbol": "AAPL",
            "curr_date": "2026-01-04",
            "atr_lookback": 20,
        })

        self.assertIn("Not enough history to compute volatility analysis", result)

if __name__ == "__main__":
    unittest.main()