"""Slack notification shim.

The neural-network training scripts (``train_nn.py`` / ``train_multi_layer_nn.py``)
import ``send_slack_notification`` from ``src.notify``. This repository has a single
Slack implementation in :mod:`src.utils`, whose credentials come from
``credentials.yaml`` (never hardcoded). This module re-exports it so both the
axis and NN experiments share one logger, one Slack client, and one config path.
"""
from src.utils import send_slack_notification

__all__ = ["send_slack_notification"]
