# Polymarket Paper Trader — Tennis & Table Tennis

A production-structured **paper trading** system for Polymarket in-play tennis and
table tennis markets. It connects to real Polymarket data, runs a real
point-level probability model, and simulates execution against the real order
book — with **no wallet, no private key, and no order-signing code anywhere in
the project**. It cannot place a real trade.

Starting bankroll is $100 of imaginary money.

---

## The strategy, stated honestly

The claim being traded is narrow, and it's worth being precise about it because
it determines whether any of this can work:

> We are **not** claiming to know the players better than the market. We take the
> market's own pre-match price as the truth about relative strength, convert it
> into serve-point probabilities, and then use the live score to compute what the
> price *should* be. The bet is that the order book is slower to reprice a score
> than the arithmetic is.

Concretely:

1. **Anchor.** When a match starts, read the market's pre-match price (say
   Sinner 0.68). Solve for the pair of serve-point probabilities `(pa, pb)` that
   reproduce exactly 0.68 from a cold start.
2. **Reprice.** Every time the score changes, feed the new score into the Markov
   model. It returns the exact win probability implied by the market's own
   opinion plus the score.
3. **Compare.** If the model says 0.74 and the book is still offering 0.70, the
   book hasn't caught up. That gap is the trade.
4. **Decay.** The anchor is pulled toward the market over time. If the market
   persistently disagrees, it usually knows something the model doesn't — an
   injury, a medical timeout, a retirement. Fighting that with a Markov chain is
   how you lose money confidently.

The edge is real but small, and it decays in seconds. Everything in the risk
layer exists to stop a $100 account from dying while trying to harvest it.

---

## Quick start

```bash
git clone <your-repo-url>
cd polymarket_paper_trader

# Simulation and tests need nothing installed.
python run.py simulate --matches 200

# Live paper trading needs two packages.
pip install -r requirements.txt
python run.py live --minutes 60

# Watch it in a browser (separate terminal).
python run.py dashboard      # http://127.0.0.1:8000
```

### Commands

| Command | What it does |
|---|---|
| `python run.py simulate` | Offline simulation. No network. |
| `python run.py simulate --sweep` | **The important one.** Sweeps market speed to show where the edge dies. |
| `python run.py live` | Paper trade real markets in real time. |
| `python run.py dashboard` | Web UI on `http://127.0.0.1:8000`. |
| `python run.py markets` | List currently tradeable tennis markets. |
| `python run.py report` | Print the saved portfolio. |
| `python -m unittest discover -s tests` | Run the test suite (90 tests). |

---

## Start here: the sweep

Before trusting any backtest, run this:

```bash
python run.py simulate --matches 200 --sweep
```

```
   catchup   return %   trades   signals  max dd %
  ------------------------------------------------
      0.05    1850.92      816      9418      0.41
      0.10     748.12      663      8737      1.06
      0.20     399.85      476      7144      1.41
      0.35     243.72      381      4730      0.35
      0.50      64.12      276      2838      1.89
      0.75       1.56       72      1105      5.49
      1.00       0.00        0       203      0.00
```

`catchup` is how fast the synthetic market corrects toward fair value. **The
lag is the edge.** At `1.00` the book reprices instantly, and the strategy
correctly makes *nothing* — it still generates 203 signals but executes zero
trades, because every one of them fails the net-edge test after costs.

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
  simulate.py               offline match simulator
  dashboard.py              snapshot writer + static server
  quant/
    tennis.py               point -> game -> tiebreak -> set -> match Markov model
    table_tennis.py         11-point, win-by-2, serve-every-2 model
  data/
    gamma.py                market discovery (stdlib urllib, no deps)
    feeds.py                CLOB order book WS + sports score WS, auto-reconnect
  strategy/
    live_model.py           anchoring, repricing, signal generation, exits
  execution/
    paper_broker.py         simulated fills, latency, queues, fees
    portfolio.py            cash, positions, P&L, trade journal
    risk.py                 Kelly sizing, exposure caps, kill switches
docs/
  index.html                dashboard (also the GitHub Pages site)
tests/                      90 unit tests, stdlib unittest only
```

### The models

Both sports reduce to two numbers: the probability each player wins a point on
their own serve. Everything else is derived exactly by recursion — no simulation,
no fitted curves.

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

---

## Risk controls

On a $100 bankroll what kills you isn't a bad model, it's one oversized position
in a market you misread. Every trade must survive all of:

**Sizing** — fractional Kelly (default 0.20×), capped at 10% of equity per trade,
15% per market, 60% total exposure.

**Entry filters** — minimum 3% edge *after* assumed slippage and fees; spread
under 5c; at least 50 shares of depth; book fresher than 5s; price between 0.05
and 0.95 (the tails have asymmetric resolution risk and illusory edges); signal
younger than 45s.

**Kill switches** — 25% max drawdown, 15% daily loss limit, $20 equity floor.
Once tripped, trading stops permanently for the session.

It also **refuses to trade a match it didn't see from the start**, because
without a clean pre-match anchor the calibration is guesswork.

---

## The dashboard

`python run.py dashboard` serves a live view: equity curve, open positions,
recent fills, matches being tracked with model-vs-market side by side, feed
health, and a breakdown of *why* signals were rejected — usually the most
informative panel, since it tells you which filter is doing the work.

The bot writes `state/data.json` continuously and mirrors it to `docs/data.json`.

### Publishing to GitHub Pages

`docs/index.html` reads `docs/data.json` as a relative path, so the same page
works locally and hosted with no changes.

1. Push the repo to GitHub.
2. **Settings → Pages → Source: Deploy from a branch**, branch `main`, folder
   `/docs`.
3. Your dashboard is live at `https://<username>.github.io/<repo>/`.
4. Commit `docs/data.json` whenever you want to publish a snapshot:

```bash
git add docs/data.json && git commit -m "update results" && git push
```

Note this publishes a *snapshot*, not a live feed — GitHub Pages serves static
files. For a genuinely live public dashboard you'd need a host that runs the bot
continuously.

---

## Configuration

Everything is in `config.yaml`. The knobs that matter most:

| Setting | Why it matters |
|---|---|
| `broker.latency_ms` | The single biggest realism lever. Raise it if unsure. |
| `strategy.signal_ttl_ms` | How long after a score change you believe an edge is real. |
| `strategy.model_weight` | 1.0 trusts the model fully; lower shrinks toward the market. |
| `risk.kelly_fraction` | Full Kelly on an unvalidated model empties the account. |
| `risk.min_edge` | Must exceed real costs or you're paying the spread for fun. |

---

## Honest limitations

- **The score feed is game-level, not point-level.** Polymarket's sports stream
  gives `"6-4, 3-2"`, not who's serving or the current point. Server is inferred
  from game parity — right about half the time, worth roughly one game of
  accuracy. Point-level data would materially improve this.
- **The model doesn't know about injuries, cramping, weather, or momentum.**
  That's exactly what `model_weight` and anchor decay are defending against.
- **In-play tennis is a competitive market.** Firms with colocated infrastructure
  and direct scoring feeds are trading this. A retail connection at 250ms is not
  a latency advantage, which is why the sweep matters so much.
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
5. Passive market making once the fair-value model has proven itself.

---

## Disclaimer

Educational software for studying market microstructure and probability
modelling. It is not financial advice. It cannot place real trades by design. If
you ever adapt it to trade real money, understand that prediction markets are
risky, that most retail participants lose, and that a strategy which backtests
well usually stops working the moment real fills are involved.

MIT licensed.
