"""Pith client exceptions."""


class PithError(Exception):
    """Base exception for all Pith client errors."""


class PithAPIError(PithError):
    """Server returned an error response."""

    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class PithAuthError(PithAPIError):
    """Authentication failed (401/403)."""

    def __init__(self, detail: str = "Invalid or missing API key"):
        super().__init__(401, detail)


class PithTimeoutError(PithError):
    """Request timed out."""

    def __init__(self, timeout: float, endpoint: str = ""):
        self.timeout = timeout
        self.endpoint = endpoint
        super().__init__(f"Request to {endpoint} timed out after {timeout}s")


class PithConnectionError(PithError):
    """Could not connect to the Pith server."""

    def __init__(self, url: str = "", detail: str = ""):
        self.url = url
        super().__init__(f"Connection to {url} failed: {detail}")
