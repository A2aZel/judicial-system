from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from judicial_ai import __version__
from judicial_ai.api.middleware import RequestBodyLimitMiddleware
from judicial_ai.api.routers import analyses, health, models
from judicial_ai.config import Settings, get_settings
from judicial_ai.logging_config import configure_logging
from judicial_ai.services.analysis import AnalysisService


def create_app(
    settings: Settings | None = None,
    service: AnalysisService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    service = service or AnalysisService.from_settings(settings)
    app = FastAPI(
        title="Judicial AI Structured Analysis API",
        version=__version__,
        description=(
            "Поиск аналогов по ст. 290 УК РФ и расчёт ориентира по основному наказанию. "
            "Правовые границы требуют отдельной проверки."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.analysis_service = service
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        del request
        errors = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed", "errors": errors},
        )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(analyses.router, prefix="/api/v1")

    ui_dir = Path("ui")
    static_dir = ui_dir
    index_path = ui_dir / "index.html"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        if index_path.is_file():
            return FileResponse(index_path)
        return {"service": "judicial-ai", "docs": "/docs", "health": "/api/v1/health"}

    return app
