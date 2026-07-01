"""FastAPI application for local ML serving.

Run locally after installing API dependencies:

    uvicorn ml.api.app:app --reload --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ml.api.schemas import BatchPredictionRequest, PredictionRequest, RecommendationRequest
from ml.service import get_model_service

app = FastAPI(
    title="EcoLima ML API",
    version="0.3.0",
    description="Local ML scoring API for EcoLima candidate recycling zones.",
)


@app.get("/health")
def health(model_name: str | None = None):
    service = get_model_service(model_name) if model_name else get_model_service()
    return service.health()


@app.get("/model/metadata")
def model_metadata(model_name: str | None = None):
    service = get_model_service(model_name) if model_name else get_model_service()
    return service.metadata()


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        service = get_model_service(request.model_name)
        zone_id = request.zone_id if request.zone_id is not None else request.zone.get("candidate_id", 0)
        if request.include_explanation:
            return service.explain_record(request.zone, zone_id=zone_id)
        return service.predict_records([request.zone], zone_ids=[zone_id])[0]
    except Exception as exc:  # FastAPI boundary: surface as 400 for client debugging
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    try:
        service = get_model_service(request.model_name)
        zone_ids = request.zone_ids or [z.get("candidate_id", i) for i, z in enumerate(request.zones)]
        return {"items": service.predict_records(request.zones, zone_ids=zone_ids)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recommendations")
def recommendations(request: RecommendationRequest):
    try:
        service = get_model_service(request.model_name)
        zone_ids = request.zone_ids or [z.get("candidate_id", i) for i, z in enumerate(request.zones)]
        return {"items": service.rank_records(request.zones, zone_ids=zone_ids, top_n=request.top_n)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
