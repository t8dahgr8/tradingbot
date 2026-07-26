# Polymarket Multi-Sport Paper Trader

A production-structured **paper trading** system for Polymarket in-play
match-winner markets. It supports basketball, American football, baseball,
hockey, soccer, tennis, and table tennis. Cricket, motorsports, esports, golf,
and combat sports are deliberately excluded because this score feed does not
provide enough state for an honest in-play model.

It connects to real Polymarket data and simulates execution against the real
order book, with **no wallet, no private key, and no order-signing code anywhere
in the project**. It cannot place a real trade.

Starting bankroll is $100 of imaginary money.

The default live mode is now an **HFT-style paper market maker**. It evaluates
live markets several hundred times per minute, but only sends an order when a
passive quote should be added, moved, or cancelled. Quote cycles are not fills.
Actual simulated fills still require public volume to clear the recorded queue.
This is retail-speed market making, not colocated exchange HFT.

---

## The strategy, stated honestly

The claim being traded is narrow, and it's worth being precise about it because
it determines whether any of this can work:

> We are **not** claiming to know the teams or players better than the market.
> We take the market's own pre-match price as the strength prior, then condition
> it on the live score, period, and clock. The bet is that the order book is
> briefly slower to reprice a score than the arithmetic is.

Concretely:

1. **Anchor.** Before a game starts, read the market's winner price. Every model
   is calibrated to reproduce that probability exactly at the anchor state.
2. **Reprice.** Tennis and table tennis use point/game/set recursion. Basketball,
   football, baseball, and hockey use score-margin distributions with
   sport-specific clock handling. Soccer uses a remaining-goals Poisson update
   so home, away, and draw markets are treated separately.
3. **Compare.** If the model says 0.74 and the book is still offering 0.70, the
   book hasn't caught up. That gap is the trade.
4. **Shrink.** Fair value is blended back toward the current market. If the
   market persistently disagrees, it may know something the score feed does not:
   injuries, lineups, possession, weather, or game context.

The HFT profile watches up to 140 match-winner markets and actively quotes up to
12 live matches. It posts passive bids on complementary outcomes, skews away
from inventory, and cancels every working quote when the score changes. Acquired
inventory targets a 2% net return on cost: a fee-free passive fill at 0.82 is
offered at about 0.84. Taker exits require enough extra price to cover fees.
After 30 seconds, a smaller 0.5% net scratch profit is acceptable. It still will
not manufacture action: fills need real public trade volume.

Pregame prices are refreshed after a material 1c move. That lets the anchor
absorb public information such as withdrawals and injury news without pretending
that a scraped headline is a reliable trading signal. Orders are generated only
from a fresh live score. When a delayed or resumed match is first discovered
after play has progressed, player strength is solved against both its current
score and current winner price so the score is not counted twice.

---

## Quick start

```bash
git clone <your-repo-url>
cd polymarket_paper_trader

# Simulation and tests need nothing installed.
python run.py simulate --matches 200

# Live paper trading needs two packages.
pip install -r requirements.txt
python run.py live --mode hft --minutes 60

# Watch it in a browser (separate terminal).
python run.py dashboard      # http://127.0.0.1:8000
```

### Commands

| Command | What it does |
|---|---|
| `python run.py simulate` | Offline simulation. No network. |
| `python run.py simulate --sweep` | **The important one.** Sweeps market speed to show where the edge dies. |
| `python run.py live --mode hft` | Run the passive HFT-style paper market maker. |
| `python run.py live --mode scalp` | Run the original score-lag paper scalper. |
| `python run.py dashboard` | Web UI on `http://127.0.0.1:8000`. |
| `python run.py markets` | List currently tradeable winner markets. |
| `python run.py report` | Print the saved portfolio. |
| `python -m unittest discover -s tests` | Run the full test suite. |

---

## Start here: the sweep

Before trusting any backtest, run this:

```bash
python run.py simulate --matches 200 --sweep
```

```
   catchup   return %    fills   signals  max dd %
  ------------------------------------------------
      0.05      11.63     1277      4732      2.74
      0.10      19.41     1073      4507      0.67
      0.20      21.45      684      3948      0.12
      0.35      15.90      410      3201      0.12
      0.50      13.64      240      2237      0.00
      0.75       3.27       52       839      0.00
      1.00       0.00        0       105      0.00
```

`catchup` is how fast the synthetic market corrects toward fair value. **The
lag is the edge.** These are the current 80-match tennis results with the 0.05
fee fallback. At `1.00` the book reprices instantly, and the strategy correctly
makes *nothing* — signals appear, but none pass the net-edge test after costs.
The same 80-match table-tennis sweep returned 29.22% at `0.10`, 9.50% at
`0.50`, 1.64% at `0.75`, and 0.00% at `1.00`.

That bottom row is the honesty check. If a change ever makes it profitable,
you've found a bug, not an edge.

The top rows are **not** a forecast. They describe a market far more sluggish
than real Polymarket. Liquid ATP/WTA matches are fast; obscure ITF and minor-league
table tennis are slower. Real performance lives somewhere in the lower half of
that table, and may well be zero after costs.

---

## What makes the simulation honest

The most common way a paper trader lies is assuming you get filled at the price
you saw. This one refuses:

- **Latency is enforced.** An order never executes against the book that
  triggered it. It's held for `latency_ms` (default 250ms) and filled against the
  *next* book. There's a test proving a price gap during that window costs you.
- **The book gets walked.** Marketable orders pay the real VWAP across levels and
  partially fill when depth isn't there.
- **Queue position is modelled.** Passive orders only fill once enough volume has
  traded through to clear the shares resting ahead of them.
- **Adverse selection shows up.** A resting order that the market trades through
  gets filled at its now-stale price.
- **Current fees are charged.** Live mode reads each market's fee schedule from
  Gamma. Simulation defaults to a 0.05 taker-rate curve, and the risk gate budgets
  for both the entry and exit fee before calling an edge profitable.
- **Two equity numbers.** `equity` marks at the mid; `liquidation_equity` marks at
  the bid. On thin in-play books the gap between them is often the entire paper
  profit. The report shows both and labels the second one "the honest number".

---

## Architecture

```
run.py                      CLI
pmpt/
  models.py                 OrderBook, Order, Fill, Position, Signal
  engine.py                 async event loop: feeds -> model -> risk -> broker
  config.py                 YAML config + logging
  github_live.py            public heartbeat publisher for GitHub Pages
  simulate.py               offline match simulator
  dashboard.py              snapshot writer + static server
  quant/
    tennis.py               point -> game -> tiebreak -> set -> match Markov model
    table_tennis.py         11-point, win-by-2, serve-every-2 model
    team_sports.py          team score/clock and soccer remaining-goals models
  data/
    gamma.py                market discovery (stdlib urllib, no deps)
    feeds.py                CLOB order book WS + sports score WS, auto-reconnect
  strategy/
    live_model.py           anchoring, repricing, signal generation, exits
    market_maker.py         passive quotes, inventory skew, score-change pauses
  execution/
    paper_broker.py         simulated fills, latency, queues, fees
    portfolio.py            cash, positions, P&L, trade journal
    risk.py                 Kelly sizing, exposure caps, kill switches
docs/
  index.html                dashboard (also the GitHub Pages site)
tests/                      stdlib unittest suite
```

### The models

Tennis and table tennis reduce to the probability each player wins a point on
serve. Everything above that is derived by recursion.

Tennis handles deuce/advantage in closed form, tiebreaks with the correct
`A-BB-AA-BB` serve order, best-of-3 and best-of-5, and configurable final-set
tiebreak rules. Table tennis handles 11-point games, win-by-2, serve alternating
every 2 points then every 1 at 10-10.

Verified against known closed forms and a 400k-trial Monte Carlo:

| Case | Model | Monte Carlo |
|---|---|---|
| Tiebreak, 0.70 / 0.55 | 0.7327 | 0.7339 |
| Set, 0.68 / 0.60 | 0.7466 | 0.7466 |
| TT game, 0.60 / 0.52 | 0.6526 | 0.6527 |

**A finding worth knowing:** with equal players, both the tennis tiebreak and the
table tennis game are *exactly* 50/50 regardless of who serves first — to 12
decimal places. The tiebreak's `A-BB-AA-BB` order is the Thue–Morse sequence,
which is precisely the arrangement that eliminates first-server advantage. The
tests assert this exactly, because it's a real property and not a rounding
artifact.

For basketball, football, baseball, and hockey, the model calibrates expected
final margin to the pregame market probability, replaces the elapsed portion
with the observed home-away score, and scales uncertainty by time remaining.
Recent scoring has a deliberately small, bounded influence because the run is
already represented in the score.

Soccer needs an explicit draw state. It enumerates neutral remaining-goal
probabilities, then applies the resulting likelihood change to each market's
own home, draw, or away prior. A late join is calibrated at the current score
and price so the existing lead is not counted twice.

---

## Risk controls

On a $100 bankroll what kills you isn't a bad model, it's one oversized position
in a market you misread. The two execution modes use different gates:

**Scalp sizing** — fractional Kelly (0.15×), capped at 5% of equity per trade, 6% per
market, 40% total exposure, and 12 simultaneous positions.

**Scalp entry filters** — minimum 0.8% edge *after* assumed slippage and both expected
taker fees; confidence of at least 0.45; spread under 4c; at least 20 shares of
depth; book fresher than 5s; price between 0.05 and 0.95; signal younger than
12s.

**Scalp exits** — bank the larger of 0.4c per share or 2% net ROI, take the
larger of 0.1c or 0.5% net ROI as a scratch after 10 seconds, bail when the model
edge disappears, cap holds at 90 seconds, and cut losses at 5c.

**Kill switches** — 15% max drawdown, 10% daily loss limit, $20 equity floor.
Once tripped, trading stops permanently for the session.

**HFT inventory controls** - 5% of equity per outcome token, 8% per market,
25% total maker inventory, five shares per quote, a 5c hard stop, and a
90-second hard inventory timeout. The normal target is 2% net ROI; after 30
seconds the bot may accept a 0.5% net scratch. A working inventory offer blocks
another buy on that same token, preventing self-crossing quotes.

It refuses to trade until it has either a clean pre-match anchor or a live anchor
that round-trips the current score and winner price.

---

## The dashboard

`python run.py dashboard` serves a live view: equity curve, working quotes and
queue position, maker/taker fills, quote actions, open positions, model versus
market, feed health, and an exact breakdown of why quotes were skipped.

The bot writes `state/data.json` continuously and mirrors it to `docs/data.json`.
While live paper trading is running, it also publishes that snapshot to the
isolated `live-data` branch every 30 seconds. The public page shows `LIVE` while
heartbeats are fresh and switches to `OFFLINE` within about three minutes if the
process stops or the computer loses its connection.

### Publishing to GitHub Pages

Locally, `docs/index.html` reads `state/data.json` through the dashboard server.
On GitHub Pages it reads `data.json` from the `live-data` branch. That branch is
force-updated as a single commit, keeping heartbeat noise out of `main`.

1. Push the repo to GitHub.
2. **Settings → Pages → Source: Deploy from a branch**, branch `main`, folder
   `/docs`.
3. Your dashboard is live at `https://<username>.github.io/<repo>/`.
4. Run `python run.py live`. Existing Git credentials publish the heartbeat; no
   token is embedded in the page or source code.

GitHub Pages still serves only the frontend. The Python process runs on your
computer, so the public dashboard is live only while that computer, the bot, and
its internet connection are on. Use `--no-publish-github` to keep a session local.

---

## Configuration

Everything is in `config.yaml`. The knobs that matter most:

| Setting | Why it matters |
|---|---|
| `broker.latency_ms` | The single biggest realism lever. Raise it if unsure. |
| `run.mode` | `hft` for passive market making or `scalp` for the original taker strategy. |
| `market_maker.quote_refresh_ms` | Minimum interval between quote reconciliations per market. |
| `market_maker.min_quote_edge` | Required fair-value cushion for a passive bid. |
| `market_maker.min_exit_roi` | Normal net return-on-cost target for maker inventory. |
| `market_maker.scratch_exit_roi` | Smaller net target accepted after the soft inventory age. |
| `market_maker.max_total_inventory_pct` | Portfolio-wide maker inventory ceiling. |
| `market_maker.hard_inventory_age_ms` | Maximum time to carry unclosed maker inventory. |
| `strategy.signal_ttl_ms` | How long after a score change the scalp mode considers an edge live. |
| `strategy.model_weight` | 1.0 trusts the model fully; lower shrinks toward the market. |
| `strategy.pregame_reanchor_threshold` | Refreshes the baseline after a meaningful pregame line move. |
| `strategy.quick_take_profit` | Bid-side profit threshold for taking a fast scalp. |
| `strategy.quick_take_profit_roi` | Percentage return threshold applied alongside the fixed amount. |
| `strategy.scratch_profit` | Tiny profit target after the trade has been open briefly. |
| `broker.passive_entries` | Passive mode is available, but disabled after adverse-selection stress tests. |
| `risk.kelly_fraction` | Full Kelly on an unvalidated model empties the account. |
| `risk.min_edge` | Remaining edge required after slippage and expected fees. |

---

## Honest limitations

- **The public score feed can be delayed, wrong, or incomplete.** The team models
  use only home-away score, period, and clock. They do not see possession,
  timeouts, lineups, cards, pitching changes, weather, or play-by-play quality.
- **Tennis is game-level, not point-level.** The feed gives `"6-4, 3-2"`, not
  who is serving or the current point. Server is inferred from game parity.
- **Momentum is not treated as magic.** A recent scoring run gets a small bounded
  adjustment in team sports, because its points are already in the score and
  large hot-hand adjustments would double-count them.
- **In-play tennis is a competitive market.** Firms with colocated infrastructure
  and direct scoring feeds are trading this. A retail connection at 250ms is not
  a latency advantage, which is why the sweep matters so much.
- **Fast quote evaluation does not create liquidity.** A paper order can remain
  unfilled for minutes because it is behind real volume. Forcing marketable churn
  would mainly add taker fees and adverse selection.
- **Table tennis minor leagues have a documented match-fixing history.** Unusual
  pre-match volume is a red flag, and this system does not currently detect it.
- **Paper trading omits the hardest part**: real fills change the book, and real
  losses change your behaviour.

---

## Roadmap, roughly in order of value

1. Log every signal with its outcome, so realized edge can be measured against
   predicted edge. Without this you're guessing.
2. Point-level score data from a dedicated provider.
3. Player-specific serve statistics instead of a surface-average anchor.
4. Cross-market arbitrage against sportsbook prices.
5. Cancel-latency and order-rate-limit modelling for a stricter maker stress test.

---

## Disclaimer

Educational software for studying market microstructure and probability
modelling. It is not financial advice. It cannot place real trades by design. If
you ever adapt it to trade real money, understand that prediction markets are
risky, that most retail participants lose, and that a strategy which backtests
well usually stops working the moment real fills are involved.

MIT licensed.
