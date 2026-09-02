"""
Will a municipality grow or shrink?

A binary classifier over Finland's 562 municipalities: given a municipality's
demographic and economic profile in a base year, predict whether its
population is larger or smaller five years later.

This is the question behind most Finnish municipal policy arguments — school
closures, service centralisation, the wellbeing-services-county funding
formula — so it is worth predicting honestly.

Two things the implementation is careful about:

*No leakage.* The obvious feature, "population change over the last five
years", is excluded. It is nearly the label shifted backwards, and including
it produces a model that looks excellent and has learned nothing. Features are
restricted to the level and structure of a municipality at the base year.

*A real baseline.* Finnish municipalities are overwhelmingly shrinking, so
"predict shrink for everyone" is already a strong classifier. Accuracy is
reported next to that majority-class rate, and balanced accuracy next to it
again, because on imbalanced data raw accuracy flatters everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .statfin import WHOLE_COUNTRY, StatFinClient

log = logging.getLogger(__name__)

# Level-and-structure features only. Nothing describing recent population
# change, which would leak the label.
FEATURES: tuple[str, ...] = (
    "Väkiluku",
    "Demografinen huoltosuhde",
    "Taloudellinen huoltosuhde",
    "Työttömien osuus työvoimasta, %",
    "Työpaikkaomavaraisuus",
    "Asuntokuntien keskikoko, henkilöä",
    "Syntyneiden enemmyys, henkilöä",
)

LABEL = "Väkiluku"


@dataclass(frozen=True)
class TrainingReport:
    """What the model scored, and what it had to beat."""

    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    majority_baseline: float
    base_year: int
    horizon: int
    positive_rate: float

    @property
    def beats_baseline(self) -> bool:
        return self.accuracy > self.majority_baseline

    def summary(self) -> str:
        verdict = "beats" if self.beats_baseline else "does not beat"
        return (
            f"accuracy {self.accuracy:.1%} ({verdict} the {self.majority_baseline:.1%} "
            f"majority baseline), balanced accuracy {self.balanced_accuracy:.1%}, "
            f"ROC AUC {self.roc_auc:.3f}, n={self.n_train}+{self.n_test}"
        )


@dataclass
class GrowthClassifier:
    """Predicts population growth over `horizon` years from a base-year profile."""

    base_year: int = 2018
    horizon: int = 5
    pipeline: Pipeline | None = None
    report: TrainingReport | None = None
    feature_names: tuple[str, ...] = FEATURES

    def _build_dataset(self, client: StatFinClient) -> tuple[np.ndarray, np.ndarray, list[str]]:
        columns: dict[str, dict[str, float]] = {
            name: client.cross_section(name, self.base_year) for name in self.feature_names
        }
        start = client.cross_section(LABEL, self.base_year)
        end = client.cross_section(LABEL, self.base_year + self.horizon)

        # KOKO MAA is the national total, not a municipality — including it
        # would put an enormous outlier in the training set.
        areas = sorted(
            area for area in start if area != WHOLE_COUNTRY and area in end and start[area] > 0
        )

        rows: list[list[float]] = []
        labels: list[int] = []
        kept: list[str] = []
        for area in areas:
            rows.append([columns[name].get(area, np.nan) for name in self.feature_names])
            labels.append(int(end[area] > start[area]))
            kept.append(area)

        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), kept

    def fit(self, client: StatFinClient, random_state: int = 0) -> TrainingReport:
        features, labels, areas = self._build_dataset(client)
        if len(set(labels.tolist())) < 2:
            raise ValueError("only one class present — cannot train a classifier")

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=random_state, stratify=labels
        )

        pipeline = Pipeline(
            [
                # Municipalities occasionally miss an indicator in a given year;
                # median imputation keeps the row rather than dropping the town.
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        )
        pipeline.fit(x_train, y_train)

        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        majority = float(max(np.mean(y_test), 1 - np.mean(y_test)))
        report = TrainingReport(
            n_train=len(y_train),
            n_test=len(y_test),
            accuracy=float(np.mean(predicted == y_test)),
            balanced_accuracy=float(balanced_accuracy_score(y_test, predicted)),
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            majority_baseline=majority,
            base_year=self.base_year,
            horizon=self.horizon,
            positive_rate=float(np.mean(labels)),
        )

        self.pipeline = pipeline
        self.report = report
        log.info("trained on %d municipalities: %s", len(areas), report.summary())
        return report

    def predict_proba(self, profile: dict[str, float]) -> float:
        """Probability that a municipality with this profile grows."""
        if self.pipeline is None:
            raise RuntimeError("classifier is not trained — call fit() first")
        row = np.asarray([[profile.get(name, np.nan) for name in self.feature_names]], dtype=float)
        return float(self.pipeline.predict_proba(row)[0, 1])

    def importances(self) -> dict[str, float]:
        """Feature importances, most important first."""
        if self.pipeline is None:
            raise RuntimeError("classifier is not trained — call fit() first")
        model = self.pipeline.named_steps["model"]
        pairs = zip(self.feature_names, model.feature_importances_, strict=True)
        return dict(sorted(pairs, key=lambda kv: kv[1], reverse=True))
