import secrets

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from judicial_ai.services.analysis import AnalysisService

bearer = HTTPBearer(auto_error=False)


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    configured = request.app.state.settings.api_key
    if configured is None:
        return
    expected = configured.get_secret_value().encode("utf-8")
    supplied = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else ""
    if not secrets.compare_digest(supplied.encode("utf-8"), expected):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
