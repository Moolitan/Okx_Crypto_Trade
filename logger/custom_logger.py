# -*- coding: utf-8 -*-
"""
Custom logger utility.

Provides a unified logging interface for consistent console output
across different modules.
"""

import time
from typing import Optional


class CustomLogger:
    """
    Custom logger class.

    Provides a simple, unified logging interface that can be reused
    across different modules.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            name: Optional logger name used to identify the log source.
        """
        self.name = name

    def log(self, message: str):
        """
        Core logging method.

        Args:
            message: Log message to output.
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        if self.name:
            print(f"[{timestamp}] [{self.name}] {message}", flush=True)
        else:
            print(f"[{timestamp}] {message}", flush=True)

    def info(self, message: str):
        """
        Info-level log.
        """
        self.log(f"ℹ️ {message}")

    def warning(self, message: str):
        """
        Warning-level log.
        """
        self.log(f"⚠️ {message}")

    def error(self, message: str):
        """
        Error-level log.
        """
        self.log(f"❌ {message}")

    def success(self, message: str):
        """
        Success-level log.
        """
        self.log(f"✅ {message}")

    def debug(self, message: str):
        """
        Debug-level log.
        """
        self.log(f"🔍 {message}")


# Global default logger instance for convenience
_default_logger = CustomLogger()


def log(message: str):
    """
    Global logging function (backward compatible).

    Args:
        message: Log message to output.
    """
    _default_logger.log(message)
