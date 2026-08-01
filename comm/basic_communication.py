"""Deterministic Chapter-3 task handoff service.

This is the only communication implementation retained by the Chapter-3
project.  It has no physical-channel model or random-number-generator
dependency and models exactly one reliable target-coordinate delivery after a
fixed integer delay.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


def _copy_target(target: Any) -> Any:
    """Return an owned copy without imposing a NumPy or Torch dependency."""
    detach = getattr(target, "detach", None)
    clone = getattr(target, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    copy_method = getattr(target, "copy", None)
    if callable(copy_method):
        return copy_method()
    return deepcopy(target)


class FixedReliableHandoff:
    """One-shot, lossless, fixed-delay target-coordinate handoff."""

    model_id = "fixed_reliable_one_step_v1"

    def __init__(self, delay_steps: int = 1):
        delay_steps = int(delay_steps)
        if delay_steps < 1:
            raise ValueError("delay_steps must be at least one")
        self.delay_steps = delay_steps
        self.reset()

    def reset(self) -> None:
        self._event: Optional[Dict[str, Any]] = None
        self._delivery_emitted = False

    def publish_target(self, *, found_step: int, finder_idx: int, target: Any) -> bool:
        """Register the episode's only handoff event.

        Returns ``True`` only for the first publication.  Duplicate discoveries
        cannot replace the original target or change its delivery time.
        """
        if self._event is not None:
            return False
        found_step = int(found_step)
        self._event = {
            "found_step": found_step,
            "handoff_step": found_step,
            "delivery_step": found_step + self.delay_steps,
            "finder_idx": int(finder_idx),
            "target": _copy_target(target),
            "delivered": False,
        }
        return True

    def advance(self, *, entering_step: int) -> Optional[Dict[str, Any]]:
        """Emit one copied event when its deterministic delivery step is due."""
        if self._event is None or self._delivery_emitted:
            return None
        if int(entering_step) < int(self._event["delivery_step"]):
            return None
        self._event["delivered"] = True
        self._delivery_emitted = True
        return self.state_dict()

    def state_dict(self) -> Optional[Dict[str, Any]]:
        if self._event is None:
            return None
        state = dict(self._event)
        state["target"] = _copy_target(self._event["target"])
        return state
