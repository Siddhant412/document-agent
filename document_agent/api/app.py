from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from document_agent.api.routes import router
from document_agent.config import Settings, get_settings
from document_agent.db.connection import close_pool, init_db
from document_agent.logging_config import configure_logging
from document_agent.storage import ObjectStore

_AUTH_EXEMPT_PATHS = {"/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings)
    ObjectStore(settings).ensure_bucket()
    yield
    close_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="Document Agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def api_key_auth(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        settings = get_settings()
        if _is_request_authorized(request, settings):
            return await call_next(request)
        return JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)

    app.include_router(router)
    return app


app = create_app()


def _is_request_authorized(request: Request, settings: Settings) -> bool:
    if not settings.api_key:
        return True
    path = request.url.path.rstrip("/") or "/"
    if path in _AUTH_EXEMPT_PATHS:
        return True
    configured = settings.api_key
    supplied = request.headers.get(settings.api_key_header)
    authorization = request.headers.get("Authorization", "")
    if not supplied and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1].strip()
    return bool(supplied and secrets.compare_digest(supplied, configured))
