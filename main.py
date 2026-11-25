#!/usr/bin/env python3
"""Tiny Bot, Easy $2: EMA-tilt rotation bot for Solana hourly bars (v1).

Strategy (written from scratch for this bounty, not a copy of any kit
baseline):

- Every symbol keeps its hourly closes. From them we maintain a fast EMA
  (6 bars) and a slow EMA (24 bars) plus the trailing 24-bar return.
- Every 12 bars, starting once 24 bars of history exist, each symbol is
  scored as (fast EMA / slow EMA - 1) + 0.5 * (trailing 24-bar return).
  The bot targets the top 2 symbols by that score, but only while the
  score is positive (> 0.003); otherwise it holds cash.
- A held symbol whose close falls more than 6% below the close at which
  the bot bought it is sold at the next rebalance (stop-loss).
- Sells go out first; buys follow on the next bar, because a sell's
  proceeds never fund a buy in the same reply. Each buy spends an even
  share of the cash, capped at the 25% max-weight cap, and is skipped
  when the modelled price impact is over 150 bps.
- Long only, no leverage, no shorts. Deterministic: plain float EMAs in
  universe order, no clock, no environment, no randomness, so the two
  grading runs produce byte-identical action logs.

Runs under the bars-replay/v1 protocol through the kit's Python shim,
vendored at vendor/makefun_bars.py as the kit instructs. Python 3
standard library only, no network.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

FAST_SPAN = 6
SLOW_SPAN = 24
REBALANCE_EVERY = 12
TOP_N = 2
MIN_SCORE = 0.003
STOP_LOSS = 0.06
MAX_SLIPPAGE_BPS = 2000
MAX_IMPACT_BPS = 150

# Per-symbol close history, indexed by batch seq (None before first bar).
closes = {}
# Fast/slow EMA per symbol, seeded on the first close.
ema_fast = {}
ema_slow = {}
# Close at the batch where a buy was decided, per held symbol (stop-loss).
entry_close = {}
# Target chosen on a rebalance bar whose sells are still filling.
pending_target = None
next_rebalance = SLOW_SPAN


def _ema(prev, price, span):
    alpha = 2.0 / (span + 1.0)
    return price * alpha + prev * (1.0 - alpha) if prev is not None else price


def _score(state, symbol, seq):
    hist = closes[symbol]
    now = hist[seq]
    then = hist[seq - SLOW_SPAN]
    if now is None or then is None:
        return None
    fast = ema_fast[symbol]
    slow = ema_slow[symbol]
    if slow == 0:
        return None
    return (fast / slow - 1.0) + 0.5 * (now / then - 1.0)


def _target(state, seq):
    ranked = []
    for i, item in enumerate(state.hello["symbols"]):
        symbol = item["symbol"]
        score = _score(state, symbol, seq)
        if score is not None and score > MIN_SCORE:
            ranked.append((-score, i, symbol))
    ranked.sort()
    return [symbol for _, _, symbol in ranked[:TOP_N]]


def _buys(state, symbols):
    caps = state.hello["caps"]
    fees = state.hello["fees"]
    room = min(TOP_N, caps["max_open_positions"], caps["max_orders_per_bar"]) - len(state.positions)
    coming = [s for s in symbols if s not in state.positions and state.close(s) is not None][: max(room, 0)]
    if not coming:
        return []
    orders = []
    for symbol in coming:
        even = state.cash // len(coming)
        cap = state.weight_room(symbol)
        amount = min(even, cap)
        if amount < int(fees["min_notional_quote_units"]):
            continue
        impact = state.impact_bps(symbol, amount)
        if impact is not None and impact > MAX_IMPACT_BPS:
            continue
        entry_close[symbol] = state.close(symbol)
        orders.append(
            {
                "side": "buy",
                "symbol": symbol,
                "notional_quote_units": amount,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        )
    return orders


def _sells(state, want):
    orders = []
    for symbol, holding in list(state.positions.items()):
        entry = entry_close.get(symbol)
        stopped = entry is not None and state.close(symbol) is not None and state.close(symbol) < entry * (1.0 - STOP_LOSS)
        if symbol not in want or stopped:
            entry_close.pop(symbol, None)
            orders.append(
                {
                    "side": "sell",
                    "symbol": symbol,
                    "qty": holding.qty,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }
            )
    return orders


def on_hello(state):
    global next_rebalance
    next_rebalance = max(SLOW_SPAN, state.warmup_bars)


def on_batch(state, batch):
    global pending_target, next_rebalance
    seq = batch["seq"]
    for item in state.hello["symbols"]:
        symbol = item["symbol"]
        hist = closes.setdefault(symbol, [])
        while len(hist) <= seq:
            hist.append(None)
        close = state.close(symbol)
        hist[seq] = close
        if close is not None:
            ema_fast[symbol] = _ema(ema_fast.get(symbol), float(close), FAST_SPAN)
            ema_slow[symbol] = _ema(ema_slow.get(symbol), float(close), SLOW_SPAN)
    if state.warming_up:
        return []
    if pending_target is not None:
        target = pending_target
        pending_target = None
        return _buys(state, target)
    if seq < next_rebalance:
        return []
    next_rebalance = seq + REBALANCE_EVERY
    want = _target(state, seq)
    sells = _sells(state, want)
    if sells:
        pending_target = want
        return sells
    return _buys(state, want)


makefun_bars.run(on_batch, on_hello=on_hello)
