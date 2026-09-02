"""
Grounded question answering over Finnish municipal data, via Azure OpenAI.

The model is never asked to recall a figure. It is given tools that read the
real StatFin series, and the system prompt forbids answering from memory — an
LLM that half-remembers a Finnish town's population is worse than useless in a
public-sector context, because the answer looks exactly as confident as a
correct one.

The pattern is a standard tool-calling loop:

    question -> model picks a tool -> we run it against StatFin
             -> result returned to the model -> model writes the answer

Every number in the answer therefore came out of a function call this process
made, and `Answer.sources` records them so the caller can show provenance.

When Azure credentials are absent the class degrades to `OfflineAssistant`,
which runs the same tools and formats the result without a model. Tests and
local demos then work with no subscription, and CI needs no secrets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .classify import GrowthClassifier
from .config import Settings
from .config import settings as default_settings
from .forecast import forecast as run_forecast
from .statfin import StatFinClient, StatFinError

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You answer questions about Finnish municipalities using Tilastokeskus (StatFin) data.

Rules:
- Never state a figure from memory. Every number must come from a tool call.
- If a tool fails or the data does not exist, say so plainly. Do not estimate.
- A forecast is a projection, not a fact. When you quote one, give its backtest
  error and say whether it beat the naive baseline.
- Answer in the language the user asked in (Finnish or English).
- Be concise. Give the number, its year, and what it means. No preamble.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_series",
            "description": (
                "Historical values of one indicator for one Finnish municipality, "
                "1987-2025. Use for any question about a current or past figure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {"type": "string", "description": "Municipality, e.g. 'Tampere'"},
                    "indicator": {
                        "type": "string",
                        "description": "StatFin indicator name, e.g. 'Väkiluku'",
                    },
                    "last_n_years": {"type": "integer", "description": "How many recent years"},
                },
                "required": ["area", "indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Project an indicator forward for a municipality. Returns the "
                "projection together with its walk-forward backtest error and the "
                "error of a naive baseline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {"type": "string"},
                    "indicator": {"type": "string"},
                    "horizon": {"type": "integer", "description": "Years ahead, default 5"},
                },
                "required": ["area", "indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_indicators",
            "description": (
                "Search the 265 available StatFin indicator names. Call this first "
                "when unsure of the exact name of an indicator."
            ),
            "parameters": {
                "type": "object",
                "properties": {"search": {"type": "string"}},
                "required": ["search"],
            },
        },
    },
]


@dataclass
class Answer:
    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    model: str = ""


class Tools:
    """The functions the model is allowed to call. Each returns JSON-able data."""

    def __init__(self, client: StatFinClient, classifier: GrowthClassifier | None = None) -> None:
        self.client = client
        self.classifier = classifier

    def get_series(
        self, area: str, indicator: str, last_n_years: int | None = None
    ) -> dict[str, Any]:
        series = self.client.series(area, indicator)
        records = series.as_records()
        if last_n_years:
            records = records[-last_n_years:]
        return {
            "area": series.area,
            "indicator": series.indicator,
            "observations": records,
            "source": "Tilastokeskus StatFin, Kuntien avainluvut (CC BY 4.0)",
        }

    def get_forecast(self, area: str, indicator: str, horizon: int = 5) -> dict[str, Any]:
        series = self.client.series(area, indicator)
        result = run_forecast(series, horizon=horizon)
        backtest = result.backtest
        return {
            "area": result.area,
            "indicator": result.indicator,
            "method": result.method,
            "projection": result.as_records(),
            "backtest": (
                {
                    "mape_percent": round(backtest.mape, 2),
                    "naive_baseline_mape_percent": round(backtest.baseline_mape, 2),
                    "beats_baseline": backtest.beats_baseline,
                    "folds": backtest.folds,
                }
                if backtest
                else None
            ),
            "caveat": "A projection, not a measurement.",
        }

    def list_indicators(self, search: str) -> dict[str, Any]:
        needle = search.lower()
        hits = [name for name in self.client.indicators() if needle in name.lower()]
        return {"query": search, "matches": hits[:25], "total_matching": len(hits)}

    def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, name, None)
        if handler is None or name.startswith("_"):
            return {"error": f"unknown tool {name!r}"}
        try:
            return handler(**arguments)
        except (StatFinError, ValueError, TypeError) as exc:
            # Errors go back to the model as data so it can tell the user what
            # went wrong, rather than crashing the request.
            return {"error": str(exc)}


class OfflineAssistant:
    """Runs the tools and formats the result without a language model."""

    def __init__(self, tools: Tools) -> None:
        self.tools = tools

    def ask(self, question: str) -> Answer:
        area, indicator = self._guess(question)
        payload = self.tools.get_series(area, indicator, last_n_years=5)
        observations = payload["observations"]
        if not observations:
            return Answer(text=f"No data found for {indicator} in {area}.", tool_calls=1)

        latest = observations[-1]
        first = observations[0]
        change = latest["value"] - first["value"]
        direction = "up" if change > 0 else "down" if change < 0 else "flat"
        text = (
            f"{area}, {indicator}: {latest['value']:,.0f} in {latest['year']} "
            f"({direction} {abs(change):,.0f} since {first['year']}). "
            f"[offline mode — no Azure OpenAI configured, so this is a templated answer]"
        )
        return Answer(text=text, sources=[payload], tool_calls=1, model="offline")

    def _guess(self, question: str) -> tuple[str, str]:
        """Crude area/indicator extraction. Enough for a fallback, no more."""
        lowered = question.lower()
        area = next(
            (a for a in self.tools.client.areas() if a.lower() in lowered and len(a) > 3),
            "KOKO MAA",
        )
        indicator = "Väkiluku"
        for candidate in self.tools.client.indicators():
            if candidate.lower() in lowered:
                indicator = candidate
                break
        return area, indicator


class Assistant:
    """Azure OpenAI assistant with StatFin tools."""

    MAX_ROUNDS = 5

    def __init__(
        self,
        client: StatFinClient | None = None,
        config: Settings | None = None,
        classifier: GrowthClassifier | None = None,
    ) -> None:
        self.config = config or default_settings
        self.client = client or StatFinClient(cache_dir=self.config.statfin_cache_dir)
        self.tools = Tools(self.client, classifier)
        self._offline = OfflineAssistant(self.tools)
        self._azure = None

        if self.config.azure_configured:
            from openai import AzureOpenAI

            self._azure = AzureOpenAI(
                azure_endpoint=self.config.azure_openai_endpoint,
                api_key=self.config.azure_openai_api_key,
                api_version=self.config.azure_openai_api_version,
            )
        else:
            log.warning("Azure OpenAI not configured — assistant running in offline mode")

    @property
    def online(self) -> bool:
        return self._azure is not None

    def ask(self, question: str) -> Answer:
        if self._azure is None:
            return self._offline.ask(question)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        sources: list[dict[str, Any]] = []
        calls = 0

        for _ in range(self.MAX_ROUNDS):
            response = self._azure.chat.completions.create(
                model=self.config.azure_openai_deployment,
                messages=messages,
                tools=TOOLS,
                temperature=0,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return Answer(
                    text=message.content or "",
                    sources=sources,
                    tool_calls=calls,
                    model=self.config.azure_openai_deployment,
                )

            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                calls += 1
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self.tools.run(call.function.name, arguments)
                if "error" not in result:
                    sources.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return Answer(
            text="I could not settle on an answer within the allowed number of steps.",
            sources=sources,
            tool_calls=calls,
            model=self.config.azure_openai_deployment,
        )
