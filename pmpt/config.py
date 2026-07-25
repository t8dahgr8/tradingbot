"""Configuration loading. YAML if PyYAML is present, otherwise sane defaults."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

from .execution.paper_broker import BrokerConfig
from .execution.risk import RiskConfig
from .strategy.live_model import StrategyConfig


@dataclass
class RunConfig:
    starting_cash: float = 100.0
    sports: list[str] = field(default_factory=lambda: ["tennis"])
    # Only trade matches that are actually in progress. Pre-match markets are
    # where the sharp money is and where this strategy has no edge at all.
    live_only: bool = True
    discovery_interval_s: int = 180
    mark_interval_s: int = 15
    status_interval_s: int = 60
    state_dir: str = "state"
    log_level: str = "INFO"
    max_runtime_s: int = 0          # 0 = run until stopped
    max_tracked_markets: int = 40


@dataclass
class AppConfig:
    run: RunConfig = field(default_factory=RunConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def to_dict(self) -> dict:
        return {
            "run": asdict(self.run),
            "broker": asdict(self.broker),
            "risk": asdict(self.risk),
            "strategy": asdict(self.strategy),
        }


def _apply(obj: Any, data: dict | None) -> Any:
    """Overlay a dict onto a dataclass, ignoring unknown keys with a warning."""
    if not data:
        return obj
    known = {f.name for f in fields(obj)} if is_dataclass(obj) else set()
    for k, v in data.items():
        if k in known:
            setattr(obj, k, v)
        else:
            logging.getLogger(__name__).warning(
                "ignoring unknown config key %r for %s", k, type(obj).__name__
            )
    return obj


def load_config(path: str | None = None) -> AppConfig:
    cfg = AppConfig()
    if not path or not os.path.exists(path):
        return cfg
    try:
        import yaml  # type: ignore
    except ImportError:
        logging.getLogger(__name__).warning(
            "PyYAML not installed; ignoring %s and using defaults", path
        )
        return cfg
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    _apply(cfg.run, data.get("run"))
    _apply(cfg.broker, data.get("broker"))
    _apply(cfg.risk, data.get("risk"))
    _apply(cfg.strategy, data.get("strategy"))

    # Keep the fee assumption used for sizing consistent with the one the broker
    # actually charges. Silent disagreement here quietly inflates paper returns.
    cfg.risk.taker_fee_rate = cfg.broker.taker_fee_rate
    return cfg


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "trader.log"), maxBytes=5_000_000, backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s %(message)s"
        ))
        root.addHandler(fh)

    logging.getLogger("websockets").setLevel(logging.WARNING)
