"""Compatibility adapter for the merged Chapter-3 environment."""

import inspect

from env import UAVEnv as UnifiedUAVEnv
from env import _BaseUAVEnv


class UAVEnv(UnifiedUAVEnv):
    """Legacy pilot/v2/v3 adapter backed by the unified environment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


UAVEnv.__init__.__signature__ = inspect.signature(_BaseUAVEnv.__init__)

__all__ = ["UAVEnv"]
