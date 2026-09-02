"""Tool-layer and offline-fallback tests. No Azure subscription needed."""

from __future__ import annotations

import pytest

from kuntadata.assistant import Assistant, OfflineAssistant, Tools
from kuntadata.config import Settings
from kuntadata.statfin import Series, StatFinError


class FakeClient:
    """Stands in for StatFinClient so the tests never touch the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def areas(self) -> list[str]:
        return ["KOKO MAA", "Tampere", "Kajaani"]

    def indicators(self) -> list[str]:
        return ["Väkiluku", "Demografinen huoltosuhde", "Työttömien osuus työvoimasta, %"]

    def series(self, area: str, indicator: str) -> Series:
        self.calls.append((area, indicator))
        if area == "Atlantis":
            raise StatFinError("'Atlantis' is not a valid value for 'Alue'")
        return Series(
            area=area,
            indicator=indicator,
            years=list(range(2000, 2025)),
            values=[100.0 + 5 * i for i in range(25)],
        )


@pytest.fixture
def tools() -> Tools:
    return Tools(FakeClient())


def test_get_series_returns_observations_and_a_source(tools):
    result = tools.get_series("Tampere", "Väkiluku", last_n_years=3)
    assert len(result["observations"]) == 3
    assert result["observations"][-1]["year"] == 2024
    assert "StatFin" in result["source"]


def test_get_forecast_includes_its_backtest(tools):
    result = tools.get_forecast("Tampere", "Väkiluku", horizon=3)
    assert len(result["projection"]) == 3
    assert result["backtest"]["folds"] > 0
    assert "naive_baseline_mape_percent" in result["backtest"]
    assert result["caveat"]


def test_list_indicators_searches_case_insensitively(tools):
    assert "Väkiluku" in tools.list_indicators("väki")["matches"]


def test_tool_errors_come_back_as_data_not_exceptions(tools):
    """The model needs to see the failure so it can tell the user."""
    result = tools.run("get_series", {"area": "Atlantis", "indicator": "Väkiluku"})
    assert "error" in result
    assert "Atlantis" in result["error"]


def test_unknown_tool_is_reported_rather_than_raising(tools):
    assert "error" in tools.run("drop_database", {})


def test_private_attributes_are_not_callable_as_tools(tools):
    assert "error" in tools.run("_cache_path", {})


def test_offline_assistant_answers_without_a_model(tools):
    answer = OfflineAssistant(tools).ask("Kuinka paljon Tampereella on asukkaita?")
    assert "Tampere" in answer.text
    assert answer.model == "offline"
    assert answer.sources


def test_assistant_falls_back_when_azure_is_not_configured():
    assistant = Assistant(
        client=FakeClient(),
        config=Settings(azure_openai_endpoint="", azure_openai_api_key=""),
    )
    assert assistant.online is False
    assert "Tampere" in assistant.ask("Tampere Väkiluku?").text
