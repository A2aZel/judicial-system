from fastapi import APIRouter, Depends

from judicial_ai.api.dependencies import get_analysis_service, require_api_key
from judicial_ai.schemas.analysis import ModelInfo
from judicial_ai.services.analysis import AnalysisService

router = APIRouter(tags=["models"], dependencies=[Depends(require_api_key)])


@router.get("/models", response_model=ModelInfo)
def models(service: AnalysisService = Depends(get_analysis_service)) -> ModelInfo:
    return service.model_info()
