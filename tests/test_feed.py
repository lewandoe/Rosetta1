"""
tests/test_feed.py — Tests for data/feed.py and data/history.py.

No network calls — yfinance and broker are fully mocked.

Run:
    pytest tests/test_feed.py -v
"""

from __future__ import annotations

import time
import threading
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from broker.base import BrokerError, Quote
from data.feed import FeedManager, FeedError, MAX_CONSECUTIVE_ERRORS
from data.history import (
    HistoryError,
    OHLCV_COLS,
    fetch,
    fetch_multi,
    seed_bars,
    validate_sufficient_history,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quote(symbol: str = "SPY", last: float = 100.0) -> Quote:
    return Quote(
        symbol=symbol,
        bid=last - 0.05,
        ask=last + 0.05,
        last=last,
        volume=1_000_000,
        timestamp=datetime.utcnow(),
    )


def _make_ohlcv_df(symbol: str = "SPY", rows: int = 100, interval: str = "1m") -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with a tz-aware ET index."""
    base = datetime(2024, 1, 8, 9, 30, tzinfo=ET)
    index = pd.date_range(start=base, periods=rows, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 50_000,
        },
        index=index,
    )
    return df


def _make_mock_broker(quote_fn=None) -> MagicMock:
    broker = MagicMock()
    broker.get_quote.side_effect = quote_fn or (lambda sym: _make_quote(sym))
    return broker


# ===========================================================================
# data/history.py tests
# ===========================================================================

class TestHistoryFetch:
    def test_rejects_symbol_outside_universe(self):
        with pytest.raises(HistoryError, match="not in the trading universe"):
            fetch("FAKE", days=7)

    def test_raises_on_yfinance_exception(self):
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = RuntimeError("network error")
            with pytest.raises(HistoryError, match="yfinance fetch failed"):
                fetch("SPY", days=7)

    def test_raises_when_result_is_empty(self):
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            with pytest.raises(HistoryError, match="No data returned"):
                fetch("SPY", days=7)

    def test_returns_clean_dataframe(self):
        raw = _make_ohlcv_df()
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = fetch("SPY", days=7)
        assert list(df.columns) == OHLCV_COLS
        assert df.index.tz is not None
        assert str(df.index.tz) == "America/New_York"
        assert not df.empty

    def test_drops_nan_rows(self):
        raw = _make_ohlcv_df(rows=10)
        # Inject NaN in row 3
        raw.iloc[3, raw.columns.get_loc("Close")] = float("nan")
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = fetch("SPY", days=7)
        assert len(df) == 9  # one row dropped

    def test_all_nan_rows_raises(self):
        raw = _make_ohlcv_df(rows=3).astype(float)  # float dtype accepts NaN
        raw[:] = float("nan")
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            with pytest.raises(HistoryError, match="All rows dropped"):
                fetch("SPY", days=7)

    def test_columns_are_lowercase(self):
        raw = _make_ohlcv_df()
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = fetch("SPY", days=7)
        assert all(c.islower() for c in df.columns)

    def test_sorted_ascending(self):
        raw = _make_ohlcv_df(rows=20)
        # Reverse order to confirm sort
        raw = raw.iloc[::-1]
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = fetch("SPY", days=7)
        assert df.index.is_monotonic_increasing

    def test_start_date_overrides_days(self):
        """When start= is given, days= should be ignored."""
        raw = _make_ohlcv_df()
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = fetch("SPY", days=999, start=date(2024, 1, 1))
        # Just confirm it ran without error — the specific date args are
        # passed to yfinance which we've mocked
        assert not df.empty

    def test_attrs_set_on_result(self):
        raw = _make_ohlcv_df()
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = fetch("AAPL", days=7, interval="5m")
        assert df.attrs["symbol"] == "AAPL"
        assert df.attrs["interval"] == "5m"


class TestFetchMulti:
    def test_returns_dict_keyed_by_symbol(self):
        raw = _make_ohlcv_df()
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            result = fetch_multi(symbols=["SPY", "AAPL"], days=7)
        assert "SPY" in result
        assert "AAPL" in result

    def test_skips_failed_symbol_logs_error(self, caplog):
        raw = _make_ohlcv_df()

        def _history(start, end, interval, auto_adjust, prepost):
            # Fail for TSLA, succeed for others
            if "TSLA" in str(start):  # can't easily detect symbol here;
                pass                  # use side_effect approach instead
            return raw

        call_count = [0]

        def _side_effect(start, end, interval, auto_adjust, prepost):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("forced failure")
            return raw

        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = _side_effect
            import logging
            with caplog.at_level(logging.ERROR, logger="data.history"):
                result = fetch_multi(symbols=["TSLA", "SPY"], days=7)
        # TSLA failed, SPY succeeded
        assert "TSLA" not in result
        assert "SPY" in result


class TestSeedBars:
    def test_seed_bars_fetches_7_day_window(self):
        raw = _make_ohlcv_df(rows=390)
        with patch("data.history.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = raw
            df = seed_bars("SPY")
        assert not df.empty
        # Should have called history with ~7 days back
        call_kwargs = mock_ticker.return_value.history.call_args
        assert "1m" in str(call_kwargs)


class TestValidateSufficientHistory:
    def test_passes_when_enough_bars(self):
        df = _make_ohlcv_df(rows=100)
        df.attrs["symbol"] = "SPY"
        df.attrs["interval"] = "1m"
        validate_sufficient_history(df, min_bars=50)  # should not raise

    def test_raises_when_too_few_bars(self):
        df = _make_ohlcv_df(rows=10)
        df.attrs["symbol"] = "SPY"
        df.attrs["interval"] = "1m"
        with pytest.raises(HistoryError, match="Insufficient history"):
            validate_sufficient_history(df, min_bars=50)


# ===========================================================================
# data/feed.py tests
# ===========================================================================

class TestFeedManagerBasic:
    def test_get_latest_returns_none_before_first_poll(self):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=999)
        assert feed.get_latest("SPY") is None

    def test_raises_for_untracked_symbol(self):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"])
        with pytest.raises(FeedError):
            feed.get_latest("FAKE")

    def test_subscribe_raises_for_untracked_symbol(self):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"])
        with pytest.raises(FeedError):
            feed.subscribe("FAKE", lambda q: None)

    def test_get_all_latest_returns_all_symbols(self):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY", "AAPL"], poll_interval=999)
        result = feed.get_all_latest()
        assert set(result.keys()) == {"SPY", "AAPL"}


class TestFeedManagerPolling:
    def test_quote_available_after_poll(self):
        """After start(), get_latest() should return a Quote within ~1 poll cycle."""
        received = threading.Event()
        quotes = []

        def on_quote(q: Quote):
            quotes.append(q)
            received.set()

        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.05)
        feed.subscribe("SPY", on_quote)
        feed.start()

        received.wait(timeout=2.0)
        feed.stop()

        assert received.is_set(), "Quote callback never fired"
        assert len(quotes) > 0
        assert quotes[0].symbol == "SPY"

    def test_get_latest_updates_over_time(self):
        """Latest quote should change as prices change."""
        prices = [100.0, 101.0, 102.0]
        call_index = [0]

        def quote_fn(sym):
            idx = min(call_index[0], len(prices) - 1)
            call_index[0] += 1
            return _make_quote(sym, last=prices[idx])

        broker = _make_mock_broker(quote_fn=quote_fn)
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.05)
        feed.start()
        time.sleep(0.3)
        feed.stop()

        q = feed.get_latest("SPY")
        assert q is not None
        # After several polls the price should be 102.0 (last in list)
        assert q.last == pytest.approx(102.0)

    def test_multiple_symbols_polled_independently(self):
        received = {sym: threading.Event() for sym in ["SPY", "AAPL"]}

        def make_cb(sym):
            def cb(q):
                received[sym].set()
            return cb

        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY", "AAPL"], poll_interval=0.05)
        for sym in ["SPY", "AAPL"]:
            feed.subscribe(sym, make_cb(sym))
        feed.start()

        for sym in ["SPY", "AAPL"]:
            received[sym].wait(timeout=2.0)
        feed.stop()

        for sym in ["SPY", "AAPL"]:
            assert received[sym].is_set(), f"{sym} never received a quote"

    def test_stop_is_idempotent(self):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.05)
        feed.start()
        feed.stop()
        feed.stop()  # second stop should not raise


class TestFeedManagerErrorHandling:
    def test_consecutive_errors_marks_degraded(self):
        """After MAX_CONSECUTIVE_ERRORS failures, symbol is marked degraded."""
        error_count = [0]

        def failing_quote(sym):
            error_count[0] += 1
            raise BrokerError("simulated failure")

        broker = _make_mock_broker(quote_fn=failing_quote)
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.02)
        feed.start()

        # Wait long enough for all errors to accumulate
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if feed.is_degraded("SPY"):
                break
            time.sleep(0.05)
        feed.stop()

        assert feed.is_degraded("SPY"), "SPY should be degraded after repeated failures"
        assert error_count[0] >= MAX_CONSECUTIVE_ERRORS

    def test_error_counter_resets_on_success(self):
        """A successful poll after errors should reset the error counter."""
        call_count = [0]

        def flaky_quote(sym):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise BrokerError("transient error")
            return _make_quote(sym)

        broker = _make_mock_broker(quote_fn=flaky_quote)
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.02)
        feed.start()
        time.sleep(0.5)
        feed.stop()

        # Should NOT be degraded — recovered before hitting the threshold
        assert not feed.is_degraded("SPY")
        assert feed.get_latest("SPY") is not None

    def test_degraded_symbols_list(self):
        broker = _make_mock_broker(quote_fn=lambda s: (_ for _ in ()).throw(BrokerError("x")))
        feed = FeedManager(broker, symbols=["SPY", "AAPL"], poll_interval=0.02)
        feed.start()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if len(feed.degraded_symbols()) == 2:
                break
            time.sleep(0.05)
        feed.stop()

        assert set(feed.degraded_symbols()) == {"SPY", "AAPL"}

    def test_callback_exception_does_not_crash_feed(self):
        """A bad subscriber must not kill the poll thread."""
        good_received = threading.Event()

        def bad_callback(q):
            raise RuntimeError("bad subscriber")

        def good_callback(q):
            good_received.set()

        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.05)
        feed.subscribe("SPY", bad_callback)
        feed.subscribe("SPY", good_callback)
        feed.start()
        good_received.wait(timeout=2.0)
        feed.stop()

        assert good_received.is_set()

    def test_unsubscribe_removes_callback(self):
        call_count = [0]

        def cb(q):
            call_count[0] += 1

        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.05)
        feed.subscribe("SPY", cb)
        feed.start()
        time.sleep(0.15)   # ~3 polls
        feed.unsubscribe("SPY", cb)
        count_at_unsub = call_count[0]
        time.sleep(0.15)   # 3 more polls — should not increase count
        feed.stop()

        assert call_count[0] == count_at_unsub, "Callback fired after unsubscribe"


class TestFeedManagerStatus:
    def test_status_contains_all_symbols(self):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY", "NVDA"], poll_interval=999)
        s = feed.status()
        assert "SPY" in s
        assert "NVDA" in s

    def test_status_reflects_quote_after_poll(self):
        received = threading.Event()
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=0.05)
        feed.subscribe("SPY", lambda q: received.set())
        feed.start()
        received.wait(timeout=2.0)
        feed.stop()

        s = feed.status()["SPY"]
        assert s["last_price"] == pytest.approx(100.0)
        assert s["degraded"] is False
        assert s["consecutive_errors"] == 0
        assert s["last_updated"] is not None


class TestFeedManagerStartStop:
    def test_double_start_logs_warning(self, caplog):
        broker = _make_mock_broker()
        feed = FeedManager(broker, symbols=["SPY"], poll_interval=999)
        import logging
        with caplog.at_level(logging.WARNING, logger="data.feed"):
            feed.start()
            feed.start()   # second start — should warn, not crash
        feed.stop()
        assert any("already running" in r.message for r in caplog.records)
