"""Security utilities: API Key authentication, rate limiting, and request validation."""

import time
import secrets
from collections import defaultdict, deque
from typing import Optional, Dict, Deque
from fastapi import Request, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings
from app.core.logging_config import logger

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> bool:
    """Validates the X-API-Key header against configured settings.API_KEY.

    If settings.API_KEY is unset or empty, authentication is bypassed (development mode).
    If settings.API_KEY is set, valid credentials are required.
    """
    configured_key = settings.API_KEY
    if not configured_key or not configured_key.strip():
        # Authentication disabled in development mode
        return True

    if not api_key:
        logger.warning("Unauthorized access attempt: Missing X-API-Key header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'X-API-Key' authentication header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key.strip(), configured_key.strip()):
        logger.warning("Unauthorized access attempt: Invalid X-API-Key supplied.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid 'X-API-Key' authentication credential.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


class InMemoryRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False) -> None:
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self._history: Dict[str, Deque[float]] = defaultdict(deque)

    def is_rate_limited(self, client_ip: str) -> bool:
        """Checks if client_ip has exceeded allowed requests in the last 60 seconds."""
        if not self.enabled:
            return False

        now = time.time()
        window_start = now - 60.0
        timestamps = self._history[client_ip]

        # Purge timestamps older than the sliding window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self.requests_per_minute:
            return True

        timestamps.append(now)
        return False


rate_limiter = InMemoryRateLimiter(
    requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    enabled=settings.RATE_LIMIT_ENABLED,
)


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency to enforce rate limiting on incoming requests."""
    if not settings.RATE_LIMIT_ENABLED:
        return

    client_ip = request.client.host if request.client else "unknown"
    if rate_limiter.is_rate_limited(client_ip):
        logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum allowable requests per minute reached.",
        )
