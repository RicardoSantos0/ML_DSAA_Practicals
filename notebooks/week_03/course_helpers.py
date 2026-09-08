"""Course toolbox for the DSAA Machine Learning practicals.

This file sits beside the notebook that imports it, and it is cumulative: each
block says which week introduced it, so a week's copy holds what that week and
every week before it needs.

It contains no machine learning. Loading, splitting, imputing, scaling,
encoding, selecting, fitting, scoring and cross-validating all stay written out
in the notebooks, where they are the thing being taught. What is here is shared
constants and small bookkeeping objects, nothing outside the standard library,
and no path of its own: where a block touches the filesystem, the notebook
hands it the folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# -- Week 3: course plot palette (validated accessible pair) ----------------

PLOT_BLUE = "#1779A8"
PLOT_ORANGE = "#C96A00"
CLASS_COLOURS = (PLOT_BLUE, PLOT_ORANGE)


# -- Week 3: cleaning log ----------------------------------------------------


@dataclass(frozen=True)
class CleaningStep:
    """One recorded cleaning decision: what, to which column, why, and how much.

    `carries` is the same decision in a form a later week can act on, sitting
    beside the prose that explains it to a reader. A step described only in
    English can be read and not applied: the plan has to be worked out again
    from the sentence, and a plan worked out again is a new decision wearing an
    old one's name. Anything JSON can hold belongs in it, and a step that
    carries nothing forward leaves it empty.
    """

    column: str
    action: str
    reason: str
    rows_affected: int
    carries: dict | None = None


class CleaningLog:
    """A written record of the cleaning decisions you made, in order.

    The log does not clean anything. You do the cleaning in the notebook, then
    record what you did and why, so the decision survives past the cell that
    made it -- and so later weeks can start from a dataset whose history is
    written down, and apply the plan it chose instead of inventing one.

    >>> log = CleaningLog("tugas")
    >>> log.record("AvgBasket", "filled with the training mean",
    ...            "missing at random; the mean keeps the column usable", 41,
    ...            carries={"fill": "mean", "columns": ["AvgBasket"]})
    >>> len(log)
    1
    >>> log.steps[0].carries["fill"]
    'mean'
    """

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.steps: list[CleaningStep] = []

    def record(
        self, column: str, action: str, reason: str, rows_affected: int,
        carries: dict | None = None,
    ) -> None:
        """Append one decision. Every field is yours to write."""
        if not reason.strip():
            raise ValueError("every cleaning step needs a stated reason")
        self.steps.append(
            CleaningStep(column, action, reason, int(rows_affected), carries)
        )

    def records(self) -> list[dict]:
        """Return the log as plain dictionaries, ready for pd.DataFrame(...)."""
        return [
            {
                "column": step.column,
                "action": step.action,
                "reason": step.reason,
                "rows_affected": step.rows_affected,
            }
            for step in self.steps
        ]

    def to_json(self, path) -> None:
        """Write the log, `carries` included, to `path`.

        JSON rather than CSV, because `carries` is nested and a CSV would
        flatten it back into the text it exists to stop being. The notebook
        hands over the path, as everywhere else in this file.
        """
        payload = {
            "dataset": self.dataset,
            "steps": [
                {
                    "column": step.column,
                    "action": step.action,
                    "reason": step.reason,
                    "rows_affected": step.rows_affected,
                    "carries": step.carries,
                }
                for step in self.steps
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "CleaningLog":
        """Read a log back, so a later week can apply what Week 3 decided."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        log = cls(payload["dataset"])
        for step in payload["steps"]:
            log.record(step["column"], step["action"], step["reason"],
                       step["rows_affected"], step.get("carries"))
        return log

    def plan(self, column: str):
        """What the step for `column` carries forward, or None if it carries nothing."""
        for step in self.steps:
            if step.column == column:
                return step.carries
        return None

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"CleaningLog(dataset={self.dataset!r}, steps={len(self.steps)})"
