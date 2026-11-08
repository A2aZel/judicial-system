from fastapi import APIRouter, Depends

from judicial_ai.api.dependencies import get_analysis_service
from judicial_ai.services.analysis import AnalysisService

router = APIRouter(tags=["system"])


@router.get("/health", summary="Service health without secrets")
def health(service: AnalysisService = Depends(get_analysis_service)) -> dict[str, object]:
    return service.health()
