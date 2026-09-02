"""
Client for Tilastokeskus StatFin (PxWeb API v1).

The table this project uses is `Kuntien_avainluvut__uusin` — "Alueaikasarjat",
regional time series with three dimensions:

    Alue (area)        562 values   every Finnish municipality + KOKO MAA
    Tiedot (indicator) 265 values   population, land area, employment rate, ...
    Vuosi (year)        39 values   1987 - 2025

Roughly 5.8 million cells, which is what makes real forecasting and
classification possible here rather than a toy demo.

Two design notes:

*Codes are never hardcoded.* Tilastokeskus renames variable codes when the
municipal division changes — the area dimension is literally called
`alue_23_20250101`, and that string will change. Everything is resolved from
the table metadata by human-readable name at runtime, so a rename costs a
cache refresh rather than a code change.

*Responses are cached on disk.* The API is a public good with no rate limit
published; hammering it during a test run would be rude and slow. The cache
also lets the whole test suite run offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/Kuntien_avainluvut/Kuntien_avainluvut__uusin/"

WHOLE_COUNTRY = "KOKO MAA"

# Dimensions are matched by these names rather than by code — see module docstring.
_AREA_DIM = "alue"
_INDICATOR_DIM = "tiedot"
_YEAR_DIM = "vuosi"


class StatFinError(RuntimeError):
    """Raised when StatFin returns something we cannot use."""


@dataclass(frozen=True)
class Series:
    """One indicator, one area, ordered by year with gaps dropped."""

    area: str
    indicator: str
    years: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.years)

    @property
    def latest(self) -> tuple[int, float] | None:
        return (self.years[-1], self.values[-1]) if self.years else None

    def as_records(self) -> list[dict[str, float | int]]:
        return [{"year": y, "value": v} for y, v in zip(self.years, self.values, strict=True)]


class StatFinClient:
    """Thin, cached PxWeb client scoped to the municipal key-figures table."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        cache_dir: Path | str = ".cache/statfin",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._metadata: dict[str, Any] | None = None

    # ---------------------------------------------------------------- cache

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def _cached(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A truncated cache file should not be fatal; treat it as a miss.
                log.warning("discarding corrupt cache entry %s", path)
                path.unlink(missing_ok=True)
        return None

    def _store(self, key: str, payload: Any) -> None:
        self._cache_path(key).write_text(json.dumps(payload), encoding="utf-8")

    # ------------------------------------------------------------- metadata

    def metadata(self) -> dict[str, Any]:
        """Table metadata: the three dimensions and every legal value."""
        if self._metadata is not None:
            return self._metadata

        cached = self._cached("META")
        if cached is None:
            log.info("fetching StatFin table metadata")
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(self.base_url)
                resp.raise_for_status()
                cached = resp.json()
            self._store("META", cached)

        if "variables" not in cached:
            raise StatFinError("metadata has no 'variables' — table layout changed")

        self._metadata = cached
        return cached

    def _dimension(self, name_fragment: str) -> dict[str, Any]:
        """Find a dimension by a fragment of its code or label, case-insensitively."""
        for var in self.metadata()["variables"]:
            haystack = f"{var.get('code', '')} {var.get('text', '')}".lower()
            if name_fragment in haystack:
                return var
        raise StatFinError(f"no dimension matching {name_fragment!r}")

    def areas(self) -> list[str]:
        return list(self._dimension(_AREA_DIM)["valueTexts"])

    def indicators(self) -> list[str]:
        return list(self._dimension(_INDICATOR_DIM)["valueTexts"])

    def years(self) -> list[int]:
        return [int(y) for y in self._dimension(_YEAR_DIM)["valueTexts"]]

    def _code_for(self, dim_fragment: str, label: str) -> str:
        """Translate a human-readable value ('Tampere') into its API code ('KU837')."""
        dim = self._dimension(dim_fragment)
        try:
            return dim["values"][dim["valueTexts"].index(label)]
        except ValueError as exc:
            raise StatFinError(
                f"{label!r} is not a valid value for {dim.get('text', dim_fragment)!r}"
            ) from exc

    # ----------------------------------------------------------------- data

    def _post(self, query: dict[str, Any]) -> dict[str, Any]:
        key = json.dumps(query, sort_keys=True, ensure_ascii=False)
        cached = self._cached(key)
        if cached is not None:
            return cached

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.base_url, json=query)
            if resp.status_code >= 400:
                raise StatFinError(f"StatFin returned {resp.status_code}: {resp.text[:200]}")
            payload = resp.json()

        self._store(key, payload)
        return payload

    def series(self, area: str, indicator: str) -> Series:
        """Full time series for one area and one indicator, missing years dropped."""
        meta = self.metadata()
        area_dim = self._dimension(_AREA_DIM)["code"]
        ind_dim = self._dimension(_INDICATOR_DIM)["code"]
        year_dim = self._dimension(_YEAR_DIM)["code"]

        payload = self._post(
            {
                "query": [
                    {
                        "code": area_dim,
                        "selection": {
                            "filter": "item",
                            "values": [self._code_for(_AREA_DIM, area)],
                        },
                    },
                    {
                        "code": ind_dim,
                        "selection": {
                            "filter": "item",
                            "values": [self._code_for(_INDICATOR_DIM, indicator)],
                        },
                    },
                    {"code": year_dim, "selection": {"filter": "all", "values": ["*"]}},
                ],
                "response": {"format": "json-stat2"},
            }
        )
        del meta  # only needed to trigger the metadata fetch above

        year_labels = list(payload["dimension"][year_dim]["category"]["index"].keys())
        raw = payload.get("value") or []

        years: list[int] = []
        values: list[float] = []
        for label, value in zip(year_labels, raw, strict=False):
            # PxWeb uses null for "not collected that year". Dropping rather than
            # interpolating keeps the gap visible to the forecaster.
            if value is None:
                continue
            years.append(int(label))
            values.append(float(value))

        return Series(area=area, indicator=indicator, years=years, values=values)

    def cross_section(self, indicator: str, year: int) -> dict[str, float]:
        """One indicator for every municipality in a single year.

        This is the shape the classifier trains on: 562 rows, one per area.
        """
        area_dim = self._dimension(_AREA_DIM)["code"]
        ind_dim = self._dimension(_INDICATOR_DIM)["code"]
        year_dim = self._dimension(_YEAR_DIM)["code"]

        payload = self._post(
            {
                "query": [
                    {"code": area_dim, "selection": {"filter": "all", "values": ["*"]}},
                    {
                        "code": ind_dim,
                        "selection": {
                            "filter": "item",
                            "values": [self._code_for(_INDICATOR_DIM, indicator)],
                        },
                    },
                    {"code": year_dim, "selection": {"filter": "item", "values": [str(year)]}},
                ],
                "response": {"format": "json-stat2"},
            }
        )

        category = payload["dimension"][area_dim]["category"]
        labels = category["label"]
        order = category["index"]
        raw = payload.get("value") or []

        out: dict[str, float] = {}
        for code, position in order.items():
            value = raw[position] if position < len(raw) else None
            if value is not None:
                out[labels.get(code, code)] = float(value)
        return out
