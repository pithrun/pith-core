"""Run the Pith API server with platform-specific event-loop setup."""

from __future__ import annotations

import asyncio
import platform

import uvicorn


def _configure_event_loop_policy() -> None:
    if platform.system() != "Windows":
        return
    policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_factory is None:
        return
    asyncio.set_event_loop_policy(policy_factory())


def main() -> None:
    _configure_event_loop_policy()
    from app.api.server import PITH_HOST, PITH_PORT, app

    uvicorn.run(app, host=PITH_HOST, port=PITH_PORT)


if __name__ == "__main__":
    main()
