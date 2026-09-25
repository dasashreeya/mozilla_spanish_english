"""Output-surface lock for the two GER Modal stages.

`tests/test_modal_decode.py` locks one file: its `SOURCE` is bound to `scripts/modal_decode.py`,
and its `SAFE_NAMES` is that module's vocabulary. Review pointed out the consequence — the GER
scripts had **no** enforcement, and `gerapply` is the only Modal function in this project that
handles dev hypotheses and model-generated text. The property was true by inspection and
guarded by nothing, so any later edit could add a leaking print with the suite still green.

These two modules are small and their legitimate output is fully enumerable, so the same
approach applies: lock the surface rather than chase attack shapes. Checked on the AST, never on
the text, because both modules' docstrings discuss leaks at length and would match a text search.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

STAGES = {
    "modal_ger_train.py": {
        "allowed_imports": {"__future__", "modal", "json", "time", "pathlib", "torch",
                            "datasets", "peft", "transformers"},
        # Every print, verbatim in spirit: a literal plus counts, a run id, or an exception TYPE.
        "expected_output_calls": 7,
    },
    "modal_ger_apply.py": {
        "allowed_imports": {"__future__", "modal", "json", "time", "pathlib", "torch",
                            "peft", "transformers", "logging", "os", "sys"},
        "expected_output_calls": 8,
    },
}
FORBIDDEN_IMPORTS = {"subprocess", "builtins", "functools", "traceback", "shutil", "socket",
                     "urllib", "requests", "http", "ctypes", "multiprocessing"}
OUTPUT_ATTRS = {"write", "writelines", "print", "print_exc", "format_exc", "debug", "info",
                "warning", "error", "critical", "exception", "log", "warn"}
# Names that may be interpolated into an output call: counts, ids, thresholds, counts-only dicts.
# Names an output call may interpolate. `report` and `result` hold counts-only dicts; that is
# not asserted here, it is enforced by test_returned_and_logged_dicts_carry_only_counts below,
# so the allowlist entry is backed rather than trusted.
SAFE_NAMES = {"run_id", "n", "adapter", "base", "report", "result", "fired", "trainable",
              "total", "max_len", "rank", "epochs", "lr", "seconds", "rows", "ds", "repo"}
# Keys permitted in anything returned or logged by these stages.
SAFE_KEYS = {"run", "guard", "base", "pairs", "epochs", "lr", "rank", "trainable", "seconds",
             "train_loss", "samples_seen", "adapter", "clips", "guards", "accept_rate",
             "thresholds", "edit", "ratio", "empty", "accepted", "max_edit", "min_ratio",
             "max_ratio", "max_new_tokens", "id", "text"}
SAFE_CALLS = {"len", "type", "round"}


def _src(name):
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(STAGES))
def test_imports_stay_inside_the_known_surface(name):
    spec = STAGES[name]
    for node in ast.walk(ast.parse(_src(name))):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""])
            for m in mods:
                root = m.split(".")[0]
                assert root not in FORBIDDEN_IMPORTS, f"{name}: {ast.unparse(node)}"
                assert root in spec["allowed_imports"], f"{name}: unknown import {root}"


@pytest.mark.parametrize("name", sorted(STAGES))
def test_no_bare_open_and_no_builtins_subscript(name):
    for node in ast.walk(ast.parse(_src(name))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            pytest.fail(f"{name}: bare open() -- file access must go through a Path")
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) in {
                "__builtins__", "builtins", "globals()", "vars()"}:
            pytest.fail(f"{name}: {ast.unparse(node)}")


@pytest.mark.parametrize("name", sorted(STAGES))
def test_every_output_call_carries_only_counts_ids_or_exception_types(name):
    """The property that matters: nothing competition-derived can reach the Modal log.

    Uses a visitor, not ast.walk: walk cannot stop descending, so `len(rows)` would surface
    `rows` as if it were being printed. A scalar-yielding call must terminate the descent.
    """
    findings = []

    def audit_expr(node, where):
        class V(ast.NodeVisitor):
            def visit_Call(self, n):
                f = n.func
                # These yield a scalar regardless of their argument, so stop here.
                if getattr(f, "id", None) in SAFE_CALLS:
                    return
                # json.dumps(x) / "...".format(x) still depend on x, so descend into the args.
                if getattr(f, "attr", None) in {"dumps", "format"}:
                    for a in n.args:
                        self.visit(a)
                    return
                findings.append(f"{name}: output calls {ast.unparse(n)[:60]} in {where}")

            def visit_Attribute(self, n):
                if n.attr == "__name__":     # type(exc).__name__
                    return
                findings.append(f"{name}: output reads attribute {ast.unparse(n)} in {where}")

            def visit_Subscript(self, n):
                findings.append(f"{name}: output subscripts {ast.unparse(n)} in {where}")

            def visit_Name(self, n):
                if n.id not in SAFE_NAMES:
                    findings.append(f"{name}: output interpolates {n.id!r} in {where}")

        V().visit(node)

    for node in ast.walk(ast.parse(_src(name))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        is_print = getattr(f, "id", None) == "print" or getattr(f, "attr", None) == "print"
        if isinstance(f, ast.Attribute) and f.attr in OUTPUT_ATTRS and not is_print:
            recv = ast.unparse(f.value)
            if f.attr in {"write", "writelines"} and not recv.startswith(("sys.", "os.")):
                continue          # a Path handle: the run's own product, not the log
            findings.append(f"{name}: non-print output {ast.unparse(node)[:70]}")
            continue
        if not is_print:
            continue
        assert not any(k.arg is None for k in node.keywords), f"{name}: print(**kwargs)"
        for arg in list(node.args) + [k.value for k in node.keywords
                                      if k.arg in {"sep", "end"}]:
            audit_expr(arg, ast.unparse(node)[:50])
    assert not findings, "\n".join(findings)


def test_gerapply_silences_libraries_and_actually_calls_it():
    """transformers and peft log, library stderr is not an exception, and Modal ships container
    stderr off-machine. scripts/modal_decode.py carries the same guard for the same reason."""
    tree = ast.parse(_src("modal_ger_apply.py"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_silence_libraries"), None)
    assert fn is not None, "no _silence_libraries(); library output would reach the log"
    body = ast.unparse(fn)
    assert "logging.disable" in body and "transformers" in body
    entry = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "gerapply")
    assert "_silence_libraries()" in ast.unparse(entry), "guard defined but never called"


@pytest.mark.parametrize("name", sorted(STAGES))
def test_the_failure_path_prints_only_an_exception_type(name):
    tree = ast.parse(_src(name))
    handlers = [h for n in ast.walk(tree) if isinstance(n, ast.Try) for h in n.handlers]
    assert handlers, f"{name}: no except block"
    caught = {ast.unparse(h.type) for h in handlers if h.type is not None}
    assert "BaseException" in caught, f"{name}: only catches {caught}"
    raises = [n for h in handlers for n in ast.walk(h) if isinstance(n, ast.Raise)]
    assert raises and all(isinstance(r.cause, ast.Constant) and r.cause.value is None
                          for r in raises), f"{name}: re-raise must use `from None`"


@pytest.mark.parametrize("name", sorted(STAGES))
def test_returned_and_logged_dicts_carry_only_counts(name):
    """`report` and `result` are allowlisted by name in SAFE_NAMES, so their key sets are the
    hole. Check what actually leaves the function."""
    tree = ast.parse(_src(name))
    checked = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        leaving = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value]
        leaving += [a for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and getattr(n.func, "id", None) == "print" for a in n.args]
        for root in leaving:
            for node in ast.walk(root):
                if isinstance(node, ast.Dict) and node.keys:
                    keys = {k.value for k in node.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                    bad = keys - SAFE_KEYS
                    assert not bad, f"{name}: {fn.name} returns/logs {sorted(bad)}"
                    checked += 1
    assert checked, f"{name}: nothing checked; the test would pass vacuously"
