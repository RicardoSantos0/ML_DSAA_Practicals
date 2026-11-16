"""Course toolbox for the DSAA Machine Learning practicals.

Shared constants and small bookkeeping objects for the notebooks beside this
file. Each block starts in the week that first uses it, so a week's copy holds
what that week and the weeks before it need. It holds no machine learning and
nothing outside the standard library, and it names no path of its own: a block
that reads or writes a file is handed the folder by the notebook.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


# -- Week 3: course plot palette (validated accessible pair) ----------------

PLOT_BLUE = "#1779A8"
PLOT_ORANGE = "#C96A00"


# -- Week 3: cleaning log ----------------------------------------------------


@dataclass(frozen=True)
class CleaningStep:
    """One recorded cleaning decision: what, to which column, why, and how many rows.

    `carries` holds the decision in a form code can apply, anything JSON can
    hold; a step that carries nothing leaves it None.
    """

    column: str
    action: str
    reason: str
    rows_affected: int
    carries: dict | None = None


class CleaningLog:
    """The cleaning decisions made in a notebook, in order.

    The log cleans nothing: the notebook cleans, then records what it did and
    why, and `to_json` writes the record for a later notebook to `load`.
    """

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.steps: list[CleaningStep] = []

    def record(
        self, column: str, action: str, reason: str, rows_affected: int,
        carries: dict | None = None,
    ) -> None:
        """Append one decision; its reason may not be blank."""
        if not reason.strip():
            raise ValueError("every cleaning step needs a stated reason")
        self.steps.append(
            CleaningStep(column, action, reason, int(rows_affected), deepcopy(carries))
        )

    def records(self) -> list[dict]:
        """The steps as plain dictionaries, one per table row, without `carries`."""
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
        """Write the log, `carries` included, to `path`."""
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
        """Read a log written by `to_json`."""
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
                return deepcopy(step.carries)
        return None

    def __len__(self) -> int:
        return len(self.steps)


# -- Week 6: the course scoreboard -------------------------------------------
#
# A week saves its tuned result to results/week_NN.json in the folder the
# notebook names: a cross-validated and a held-out score per task, the untuned
# baseline scored the same two ways, and a fingerprint of the data, the recipe
# and the evaluated rows. Results with different fingerprints are not compared.

import hashlib

TASKS = ("classification", "regression")


def _plain_rows(rows):
    """Sorted row labels as JSON can hold them."""
    return sorted(row.item() if hasattr(row, "item") else row for row in rows)


def result_context(data_path, log_path, train_rows, test_rows, *, outer_folds=None):
    """Fingerprint the data, the recipe and the rows behind a result.

    `train_rows` and `test_rows` are the two sides of the held-out split, and
    `outer_folds` the rows each outer fold scored, in fold order.
    """
    split = json.dumps({
        "train": _plain_rows(train_rows),
        "test": _plain_rows(test_rows),
        "outer": None if outer_folds is None
        else [_plain_rows(rows) for rows in outer_folds],
    })
    return {
        "data": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "recipe": hashlib.sha256(Path(log_path).read_bytes()).hexdigest(),
        "split": hashlib.sha256(split.encode("utf-8")).hexdigest(),
    }


@dataclass(frozen=True)
class Result:
    """One model's two scores on one dataset, and the week that produced it."""

    week: int
    model: str
    cv: float
    test: float


class Scoreboard:
    """The results recorded on one dataset, each with its two scores."""

    def __init__(self, dataset: str, metric: str, floor: float | None = None,
                 *, context=None) -> None:
        self.dataset = dataset
        self.metric = metric
        self.floor = floor
        self.context = deepcopy(context)
        self.entries: list[Result] = []

    def record(self, week: int, model: str, cv: float, test: float) -> None:
        """Add one result: the cross-validated score, then the held-out one."""
        if not model.strip():
            raise ValueError("every result needs a model name")
        self.entries.append(Result(int(week), model, float(cv), float(test)))

    def save(self, results_dir, week: int, task: str, *,
             baseline_cv=None, baseline_test=None) -> None:
        """Write this week's one result on the board to results_dir/week_NN.json.

        A week that scores both tasks calls this once per task, and the two
        writes merge into one file. The baselines are the untuned model scored
        the same two ways; a week without a search leaves them None.
        """
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, not {task!r}")
        mine = [entry for entry in self.entries if entry.week == int(week)]
        if len(mine) != 1:
            raise ValueError(
                f"week {week} has {len(mine)} rows on this board; save writes "
                "exactly one headline result per week and task"
            )
        path = Path(results_dir) / f"week_{int(week):02d}.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        record["week"] = int(week)
        record[task] = {
            "model": mine[0].model,
            "metric": self.metric,
            "cv": round(float(mine[0].cv), 4),
            "test": round(float(mine[0].test), 4),
            "baseline_cv":
                None if baseline_cv is None else round(float(baseline_cv), 4),
            "baseline_test":
                None if baseline_test is None else round(float(baseline_test), 4),
        }
        if self.context is not None:
            record[task]["context"] = deepcopy(self.context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {key: record[key] for key in ["week", *TASKS] if key in record},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def table(self) -> str:
        """The board as text, in the order the results were recorded."""
        header = f"{self.dataset} -- {self.metric}"
        width = max([len(entry.model) for entry in self.entries] + [14])
        columns = (f"  {'':<{width + 9}}  {'cross-validated':>15}"
                   f"  {'held out':>9}")
        rule = "-" * max(len(header), len(columns))
        lines = [header, rule, columns]
        for entry in self.entries:
            lines.append(
                f"  week {entry.week:<2}  {entry.model:<{width}}"
                f"  {entry.cv:>15.4f}  {entry.test:>9.4f}"
            )
        if self.floor is not None:
            lines.append(rule)
            lines.append(
                f"  {'no-model floor':<{width + 9}}  {'':>15}  {self.floor:>9.4f}"
            )
        return "\n".join(lines)


# -- Week 6: the results log ------------------------------------------------
#
# One row per estimate, beside the test score of the model it produced, so the
# distance between the two is printed. The metric is passed in: F1 for one
# task, MAE for the other.


def _log_html(log, label):
    """Render scores first, moving shared evaluation context into the caption."""
    from html import escape

    context = [("frame", "CV"), ("rows", "Training rows"), ("folds", "Folds"),
               (f"validation {label} is", "Estimate")]
    columns = [("model", "Model")]
    caption = []
    for key, title in context:
        value = log[0][key]
        if all(row[key] == value for row in log):
            value = f"{value:,}" if isinstance(value, int) else str(value)
            caption.append(f"{title}: {escape(value)}")
        else:
            columns.append((key, title))
    columns += [(f"{part} {label}", f"{part.title()} {label}")
                for part in ("train", "validation", "test")]
    columns.append(("gap", "Validation − test"))
    places = 2 if label == "MAE" else 4
    header = "".join(f'<th scope="col">{escape(title)}</th>' for _, title in columns)
    body = []
    for row in log:
        cells = []
        for key, _ in columns:
            value = (row[f"validation {label}"] - row[f"test {label}"]
                     if key == "gap" else row[key])
            if key == "model":
                cells.append(f'<th scope="row"><code>{escape(str(value))}</code></th>')
            else:
                numeric = isinstance(value, (int, float))
                text = (f"{value:+,.{places}f}" if key == "gap"
                        else f"{value:,.{places}f}" if isinstance(value, float)
                        else f"{value:,}" if isinstance(value, int) else str(value))
                kind = "number" if numeric else "context"
                cells.append(f'<td class="{kind}">{escape(text)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="course-result-log">'
        '<style>'
        '.course-result-log{overflow-x:auto;margin:1em 0;color:inherit}'
        '.course-result-log table{border-collapse:collapse;width:100%;font-size:0.95em}'
        '.course-result-log caption{text-align:left;padding:0 0 0.8em;color:inherit}'
        '.course-result-log th,.course-result-log td{padding:0.65em 0.8em;'
        'border-bottom:1px solid var(--jp-border-color2,#d8d8d8)}'
        '.course-result-log thead th{text-align:right;white-space:nowrap;font-weight:600;'
        'border-bottom:2px solid var(--jp-border-color1,#aaa)}'
        '.course-result-log thead th:first-child,.course-result-log tbody th{text-align:left}'
        '.course-result-log tbody th{font-weight:400;max-width:34ch;overflow-wrap:anywhere}'
        '.course-result-log code{white-space:normal;background:transparent;color:inherit}'
        '.course-result-log .number{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}'
        '.course-result-log .context{text-align:left}'
        '.course-result-log tbody tr:nth-child(even){background:var(--jp-layout-color1,#f6f7f8)}'
        '</style><table><caption>' + ' · '.join(caption) + '</caption><thead><tr>'
        + header + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'
    )


def show_log(log, label="F1"):
    """Display an aligned notebook table, or print the full log in a terminal."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    if getattr(shell, "kernel", None) is not None:
        display(HTML(_log_html(log, label)))
        return
    rows = []
    for entry in log:
        row = {key: value for key, value in entry.items() if key != "model"}
        row["estimate minus truth"] = entry[f"validation {label}"] - entry[f"test {label}"]
        row["absolute error"] = abs(row["estimate minus truth"])
        row["model"] = entry["model"]
        rows.append(row)
    keys = list(rows[0])

    def text(key, value):
        if isinstance(value, float):
            return f"{value:+.4f}" if key == "estimate minus truth" else f"{value:.4f}"
        return str(value)

    table = [[text(key, row[key]) for key in keys] for row in rows]
    widths = [max(len(key), *(len(line[index]) for line in table))
              for index, key in enumerate(keys)]

    def line(values):
        # The model's name is the longest column, so it goes last, left-aligned.
        return "  ".join(value.ljust(width) if key == "model" else value.rjust(width)
                         for key, value, width in zip(keys, values, widths))

    print(line(keys))
    for values in table:
        print(line(values))


def record(log, result, X_test, y_test, validation_is="untuned", *, metric, label="F1"):
    """Score a returned model once on the test rows, add its row and print the log.

    `result` holds the 'frame' and 'rows' the estimate was measured on, the
    number of 'folds', the per-fold 'train' and 'validation' scores, the model's
    'name' and the fitted 'model'. `metric(y_test, predictions)` scores the test
    rows and `label` names it. Running a cell again replaces its row.
    """
    row = {
        "frame": result["frame"],
        f"validation {label} is": validation_is,
        "rows": result["rows"],
        "folds": result["folds"],
        f"train {label}": float(sum(result["train"]) / len(result["train"])),
        f"validation {label}": float(sum(result["validation"]) / len(result["validation"])),
        f"test {label}": float(metric(y_test, result["model"].predict(X_test))),
        "model": result["name"],
    }
    key = (row["frame"], validation_is, row["model"])
    log[:] = [old for old in log
              if (old["frame"], old[f"validation {label} is"], old["model"]) != key]
    log.append(row)
    show_log(log, label)


# -- Week 7: the results of earlier weeks -------------------------------------

import warnings


def results_before(week, task, results_dir, *, metric=None, context=None):
    """Every saved `task` result from a week before `week`, in week order.

    Returns (week, model, cv, test) tuples ready for Scoreboard.record. A result
    with another metric, without both scores, or, when `context` is given, with
    another fingerprint is left out with a warning.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    expected_metric = metric or {
        "classification": "F1", "regression": "MAE (EUR)"
    }[task]
    history = []
    for path in sorted(Path(results_dir).glob("week_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = record.get(task)
        if entry is None or record["week"] >= week:
            continue
        if (
            entry.get("metric") != expected_metric
            or "cv" not in entry
            or "test" not in entry
            or (context is not None and entry.get("context") != context)
        ):
            warnings.warn(
                f"Week {record['week']} {task} omitted: its metric, data, recipe "
                "or evaluated rows are different or unrecorded. Rerun that master "
                "to compare.",
                UserWarning, stacklevel=2,
            )
            continue
        history.append(
            (record["week"], entry["model"], entry["cv"], entry["test"])
        )
    return tuple(sorted(history))
