"""Semantic routing for auxiliary evidence and its admissible guarantees."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


EvidenceKind = Literal[
    "detectable_response",
    "documented_design_direction",
    "open_world_curated_edge",
]


@dataclass(frozen=True)
class EvidenceSemantics:
    """Declare what a source label means before fitting an admission gate.

    A source can be authoritative and still be open-world: a missing database
    edge is then unknown, not an error.  Such a source may be reported or used
    as a soft diagnostic, but it cannot receive a finite-sample error
    certificate whose denominator silently treats unknowns as negatives.
    """

    kind: EvidenceKind
    label_definition: str
    labels_complete_for_task: bool
    unknown_is_error: bool
    direction_scope: str

    @classmethod
    def detectable_response(cls, label_definition: str) -> "EvidenceSemantics":
        return cls(
            kind="detectable_response",
            label_definition=label_definition,
            labels_complete_for_task=True,
            unknown_is_error=True,
            direction_scope="assay-specific detectable intervention response",
        )

    @classmethod
    def documented_design(cls, label_definition: str) -> "EvidenceSemantics":
        return cls(
            kind="documented_design_direction",
            label_definition=label_definition,
            labels_complete_for_task=True,
            unknown_is_error=True,
            direction_scope="documented assignment or measurement precedence",
        )

    @classmethod
    def open_world_curated(cls, label_definition: str) -> "EvidenceSemantics":
        return cls(
            kind="open_world_curated_edge",
            label_definition=label_definition,
            labels_complete_for_task=False,
            unknown_is_error=False,
            direction_scope="curated literature/database relation",
        )

    def certificate_status(self) -> str:
        if self.labels_complete_for_task and self.unknown_is_error:
            return "eligible_for_risk_certificate"
        return "diagnostic_only_open_world"

    def validate_for_certificate(
        self,
        correct: Sequence[bool | int],
        known: Sequence[bool | int] | None = None,
    ) -> np.ndarray:
        labels = np.asarray(correct, dtype=bool)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("certificate labels must be a non-empty vector")
        if not self.labels_complete_for_task or not self.unknown_is_error:
            raise ValueError(
                "open-world evidence cannot receive a closed-world error certificate; "
                "use a diagnostic or positive-only coverage report"
            )
        if known is None:
            return labels
        mask = np.asarray(known, dtype=bool)
        if mask.shape != labels.shape or not np.all(mask):
            raise ValueError(
                "certificate fitting requires complete labels; unknown entries must "
                "be excluded by a predeclared task definition"
            )
        return labels


@dataclass(frozen=True)
class EvidenceRoute:
    """Routing decision recorded alongside every source audit."""

    semantics: EvidenceSemantics
    route: Literal["risk_controlled", "diagnostic_only"]
    reason: str


def route_evidence(semantics: EvidenceSemantics) -> EvidenceRoute:
    if semantics.certificate_status() == "eligible_for_risk_certificate":
        return EvidenceRoute(
            semantics,
            "risk_controlled",
            "task labels are complete and unknown is explicitly an error",
        )
    return EvidenceRoute(
        semantics,
        "diagnostic_only",
        "open-world source omissions are unknown and cannot define a false label",
    )
