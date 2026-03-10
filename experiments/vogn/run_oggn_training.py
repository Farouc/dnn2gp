from __future__ import annotations

from src.vogn import VOGGN

from .run_vogn_training import (
    train_classification_variational,
    train_regression_variational,
)


def train_regression_oggn(*args, **kwargs):
    return train_regression_variational(*args, optimizer_class=VOGGN, optimizer_name="OGGN", **kwargs)


def train_classification_oggn(*args, **kwargs):
    return train_classification_variational(*args, optimizer_class=VOGGN, optimizer_name="OGGN", **kwargs)

