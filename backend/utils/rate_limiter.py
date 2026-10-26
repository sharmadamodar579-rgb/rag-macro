"""
Lightweight, thread-safe in-memory rate limiter for Flask applications.
Protects sensitive endpoints (e.g., authentication, AI calls) against abuse.
"""

import time
import threading
from functools import wraps
from typing import Dict, List, Optional
from flask import request, jsonify, current_app, session


class InMemoryRateLimiter:
    """
    Sliding-window rate limiter using in-memory timestamps per client key.
    Thread-safe across multiple concurrent requests.
    """

    def __init__(self):
        self._records: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        """
        Check if an incoming request with `key` is allowed within the sliding window.

        Returns:
            tuple[bool, int]: (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            timestamps = self._records.get(key, [])
            # Filter timestamps within current window
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= max_requests:
                earliest = valid_timestamps[0]
                retry_after = max(1, int(earliest + window_seconds - now))
                self._records[key] = valid_timestamps
                return False, retry_after

            valid_timestamps.append(now)
            self._records[key] = valid_timestamps
            return True, 0

    def clear(self):
        """Clear all stored rate limit records (useful for test resets)."""
        with self._lock:
            self._records.clear()


# Global limiter instance
limiter = InMemoryRateLimiter()


def rate_limit(max_requests: int = 15, window_seconds: int = 60, key_type: str = "ip"):
    """
    Decorator to apply sliding-window rate limiting to a Flask route.

    Args:
        max_requests: Maximum allowed requests within the window.
        window_seconds: Duration of the sliding window in seconds.
        key_type: "ip" to limit by remote IP address, "user" to limit by authenticated user or IP.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Bypass rate limiting in testing mode unless explicitly testing rate limits
            if current_app.config.get("TESTING", False) and not current_app.config.get("ENABLE_RATE_LIMITS_IN_TEST", False):
                return f(*args, **kwargs)

            # Determine key
            ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1")
            if "," in ip:
                ip = ip.split(",")[0].strip()

            if key_type == "user":
                user_id = session.get("user_id")
                identifier = f"user:{user_id}" if user_id else f"ip:{ip}"
            else:
                identifier = f"ip:{ip}"

            endpoint = request.endpoint or "default"
            rate_key = f"{endpoint}:{identifier}"

            allowed, retry_after = limiter.is_allowed(rate_key, max_requests, window_seconds)
            if not allowed:
                response = jsonify({
                    "success": False,
                    "error": f"Rate limit exceeded. Too many requests. Please wait {retry_after} seconds before trying again."
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(retry_after)
                return response

            return f(*args, **kwargs)
        return wrapper
    return decorator
