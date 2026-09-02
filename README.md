# kuntadata

An AI assistant over Finnish municipal open data. Ask a question in Finnish or
English about any of Finland's municipalities and get an answer grounded in
Tilastokeskus figures, with the source cited — plus a forecast and a
classifier trained on the same data.

Built on **StatFin**, Tilastokeskus's open PxWeb API: the municipal key-figures
table holds **562 areas × 265 indicators × 39 years (1987–2025)**, about 5.8
million cells.

## What it does

| Component | Problem type | What it actually does |
|---|---|---|
| `forecast` | Time-series forecasting | Projects any municipality × indicator series forward, backtested against a naive baseline |
| `classify` | Binary classification | Predicts whether a municipality's population will grow or shrink, from its indicator profile |
| `assistant` | Generative AI (RAG) | Answers natural-language questions using retrieved StatFin figures, and cites them |

The three compose rather than sit side by side: the assistant answers "how many
people live in Tampere and what happens next" by retrieving the real series,
calling the forecaster, and explaining the result — refusing to answer when the
data does not support one.

## Running it

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
cp .env.example .env          # add your Azure OpenAI details
uv run uvicorn kuntadata.api:app --reload
```

Then <http://localhost:8000/docs>.

The forecaster and classifier need no API key — only the assistant does. It
targets **Azure OpenAI**, and falls back to a deterministic offline responder
when no credentials are configured, so the test suite and a local demo both
run without a subscription.

## Tests

```bash
uv run pytest                    # offline, uses the on-disk fixture cache
uv run pytest -m network         # additionally hits the live StatFin API
```

## Design notes

**Nothing is hardcoded against the API's variable codes.** StatFin renames them
whenever the municipal division changes — the area dimension is currently
`alue_23_20250101`, and that string has a date in it for a reason. Dimensions
and values are resolved from table metadata by human-readable name at runtime.

**Responses are cached to disk.** StatFin is a public good with no published
rate limit, and re-fetching the same series on every test run would be both
slow and rude.

**The forecaster reports its error against a baseline.** A forecast without a
backtest is a decoration. Every projection ships with its walk-forward error
and the naive-drift error next to it, so you can see whether the model earns
its complexity.

**Missing years are dropped, not interpolated.** PxWeb returns `null` for years
an indicator was not collected. Filling those in would hide a real gap from the
model and from the reader.

## Licence

MIT. StatFin data is licensed CC BY 4.0 by Tilastokeskus.
