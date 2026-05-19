import sys
import time
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from loguru import logger
from loguru._defaults import LOGURU_FORMAT

from neural_search.config import settings
from neural_search.synthesis.groq_client import GroqSynthesizer
from neural_search.api.routes import router


# ── Bridge uvicorn stdlib logging → loguru ────────────────────────────────────
class _InterceptHandler(logging.Handler):
    """
    Routes all stdlib logging (uvicorn, fastapi internals) through loguru.
    Without this, internal server errors are swallowed by uvicorn's handler
    and never appear in loguru output or log files.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _setup_logging(log_level: str = "INFO") -> None:
    """
    Remove loguru default handler, reconfigure with rotation,
    and intercept all stdlib loggers.
    """
    logger.remove()

    # Console — human readable
    logger.add(
        sys.stderr,
        format=LOGURU_FORMAT,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,   # shows variable values in tracebacks
    )

    # File — persistent, rotated daily
    settings.ensure_dirs()
    logger.add(
        settings.data_dir.parent / "logs" / "api.log",
        format=LOGURU_FORMAT,
        level="DEBUG",
        rotation="00:00",      # new file at midnight
        retention="14 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
    )

    # Intercept uvicorn, fastapi, and any other stdlib loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    settings.ensure_dirs()
    logger.info("Starting Neural Search API...")
    app.state.synthesizer = GroqSynthesizer()
    logger.info("Neural Search API ready")
    yield
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neural Search API",
    description="Hybrid semantic search over named document collections",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)


# ── Middleware: log every request + response ──────────────────────────────────
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.debug(f"→ {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as exc:
        # Unhandled exception — log full traceback before returning 500
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}\n"
            + traceback.format_exc()
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "path": str(request.url.path),
                "method": request.method,
            },
        )

    elapsed = round((time.perf_counter() - start) * 1000, 2)
    level = "warning" if response.status_code >= 400 else "debug"
    logger.log(
        level.upper(),
        f"← {response.status_code} {request.method} {request.url.path} [{elapsed}ms]",
    )
    return response


# ── Global exception handler: catches anything middleware missed ───────────────
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Exception on {request.method} {request.url.path}: {type(exc).__name__}: {exc}\n"
        + traceback.format_exc()
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {str(exc)}",
            "path": str(request.url.path),
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
