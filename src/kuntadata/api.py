"""
FastAPI surface.

The classifier is trained lazily on first use and then cached on the app
state: training pulls nine cross-sections from StatFin and fits a gradient
boosting model, which is a few seconds of work that should not happen during
import (it would make the container fail its health check on a cold start with
no network).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .assistant import Assistant
from .classify import GrowthClassifier
from .config import settings
from .forecast import MIN_POINTS
from .forecast import forecast as run_forecast
from .statfin import StatFinClient, StatFinError

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = StatFinClient(cache_dir=settings.statfin_cache_dir)
    app.state.assistant = Assistant(client=app.state.client, config=settings)
    app.state.classifier = None  # trained on first request
    yield


app = FastAPI(
    title="kuntadata",
    version="0.1.0",
    summary="Forecasting, classification and grounded Q&A over Finnish municipal open data.",
    lifespan=lifespan,
)


def get_client() -> StatFinClient:
    return app.state.client


def get_classifier() -> GrowthClassifier:
    if app.state.classifier is None:
        log.info("training growth classifier on first use")
        classifier = GrowthClassifier()
        classifier.fit(app.state.client)
        app.state.classifier = classifier
    return app.state.classifier


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["Kuinka paljon Tampereella on asukkaita?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    tool_calls: int
    model: str
    grounded: bool = Field(description="True when at least one figure came from StatFin.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "azure_openai": "configured" if settings.azure_configured else "offline-fallback",
    }


@app.get("/areas", summary="Every municipality StatFin knows about")
def areas(
    search: str | None = Query(None, description="Case-insensitive substring filter"),
    client: StatFinClient = Depends(get_client),
) -> dict[str, Any]:
    values = client.areas()
    if search:
        needle = search.lower()
        values = [a for a in values if needle in a.lower()]
    return {"count": len(values), "areas": values[:200]}


@app.get("/indicators", summary="The 265 indicators available per municipality")
def indicators(
    search: str | None = Query(None),
    client: StatFinClient = Depends(get_client),
) -> dict[str, Any]:
    values = client.indicators()
    if search:
        needle = search.lower()
        values = [i for i in values if needle in i.lower()]
    return {"count": len(values), "indicators": values[:200]}


@app.get("/series", summary="Historical series for one area and indicator")
def series(
    area: str = Query(examples=["Tampere"]),
    indicator: str = Query("Väkiluku"),
    client: StatFinClient = Depends(get_client),
) -> dict[str, Any]:
    try:
        found = client.series(area, indicator)
    except StatFinError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "area": found.area,
        "indicator": found.indicator,
        "observations": found.as_records(),
        "source": "Tilastokeskus StatFin (CC BY 4.0)",
    }


@app.get("/forecast", summary="Project a series forward, with its backtest")
def forecast_endpoint(
    area: str = Query(examples=["Tampere"]),
    indicator: str = Query("Väkiluku"),
    horizon: int = Query(5, ge=1, le=20),
    client: StatFinClient = Depends(get_client),
) -> dict[str, Any]:
    try:
        found = client.series(area, indicator)
    except StatFinError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = run_forecast(found, horizon=horizon)
    except ValueError as exc:
        # Too short to validate. A 422 is honest here: the request is
        # well-formed, the data just cannot support an answer.
        raise HTTPException(
            status_code=422,
            detail=f"{exc} (at least {MIN_POINTS} observations are required)",
        ) from exc

    backtest = result.backtest
    return {
        "area": result.area,
        "indicator": result.indicator,
        "method": result.method,
        "params": result.params,
        "projection": result.as_records(),
        "backtest": backtest
        and {
            "mape_percent": round(backtest.mape, 2),
            "naive_baseline_mape_percent": round(backtest.baseline_mape, 2),
            "skill_vs_baseline": round(backtest.skill, 3),
            "beats_baseline": backtest.beats_baseline,
            "folds": backtest.folds,
        },
        "caveat": "A projection from past trend, not a prediction of policy or migration.",
    }


@app.get("/growth", summary="Probability a municipality grows over five years")
def growth(
    area: str = Query(examples=["Kajaani"]),
    client: StatFinClient = Depends(get_client),
    classifier: GrowthClassifier = Depends(get_classifier),
) -> dict[str, Any]:
    profile: dict[str, float] = {}
    for feature in classifier.feature_names:
        try:
            profile[feature] = client.cross_section(feature, classifier.base_year)[area]
        except (StatFinError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=f"no {feature!r} for {area!r}") from exc

    report = classifier.report
    return {
        "area": area,
        "probability_of_growth": round(classifier.predict_proba(profile), 3),
        "base_year": classifier.base_year,
        "horizon_years": classifier.horizon,
        "model": {
            "accuracy": round(report.accuracy, 3),
            "balanced_accuracy": round(report.balanced_accuracy, 3),
            "roc_auc": round(report.roc_auc, 3),
            "majority_baseline": round(report.majority_baseline, 3),
            "trained_on": report.n_train + report.n_test,
        }
        if report
        else None,
        "feature_importance": {k: round(v, 3) for k, v in classifier.importances().items()},
    }


@app.post("/ask", response_model=AskResponse, summary="Ask a question in Finnish or English")
def ask(request: AskRequest) -> AskResponse:
    answer = app.state.assistant.ask(request.question)
    return AskResponse(
        answer=answer.text,
        sources=answer.sources,
        tool_calls=answer.tool_calls,
        model=answer.model or ("azure" if settings.azure_configured else "offline"),
        grounded=bool(answer.sources),
    )
