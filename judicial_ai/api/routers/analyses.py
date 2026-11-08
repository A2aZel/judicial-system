from fastapi import APIRouter, Depends

from judicial_ai.api.dependencies import get_analysis_service, require_api_key
from judicial_ai.schemas.analysis import AnalysisRequest, AnalysisResponse
from judicial_ai.services.analysis import AnalysisService

router = APIRouter(tags=["analyses"], dependencies=[Depends(require_api_key)])


@router.post("/analyses", response_model=AnalysisResponse)
def create_analysis(
    payload: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    return service.analyze(payload)
