"""Guards for the GER decode run.

These cover the three properties a pre-gpu review blocked on: the run must be launchable through
the watchdog, its log must not name a clip on any path, and its voters must be pinned to the bytes
the scored package uses.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

from scripts import modal_decode, modal_watchdog, pack_submission

SOURCE = Path(modal_decode.__file__).read_text(encoding="utf-8")


def test_turbo_is_pinned_to_the_scored_revision():
    """An unpinned Hub id resolves `main` at launch, so the corrector would train on hypotheses
    from bytes nobody scored."""
    assert modal_decode.VOTERS["turbo"]["revision"] == pack_submission.MODELS["turbo"]["revision"]


def test_every_voter_declares_where_it_lives():
    for name, spec in modal_decode.VOTERS.items():
        assert set(spec) == {"source", "on_volume", "prompted", "revision"}, name
        # A Hub model must be pinned; a volume path has no revision to pin.
        assert (spec["revision"] is None) == spec["on_volume"], name


def test_prompt_matches_the_shipped_package():
    assert modal_decode.PROMPT == pack_submission.CODE_SWITCH_PROMPT


def test_watchdog_accepts_the_decode_stage():
    stage = [a for a in _argparse_calls() if a == "stage"]
    assert stage, "watchdog lost its stage argument"
    assert "decode" in _stage_choices()


def test_watchdog_refuses_decode_under_the_default_script(monkeypatch, capsys):
    """`stage` is also the Modal function name, so decode under modal_finetune.py launches,
    fails in the container and still burns the run id."""
    monkeypatch.setattr(sys, "argv",
                        ["modal_watchdog.py", "decode", "ger1", "--max-minutes", "10"])
    with pytest.raises(SystemExit) as caught:
        modal_watchdog.main()
    assert caught.value.code == 2
    assert "scripts/modal_decode.py" in capsys.readouterr().err


def _stage_choices():
    match = re.search(r'add_argument\("stage", choices=\[([^\]]*)\]', 
                      Path(modal_watchdog.__file__).read_text(encoding="utf-8"))
    return ast.literal_eval("[" + match.group(1) + "]")


def _argparse_calls():
    text = Path(modal_watchdog.__file__).read_text(encoding="utf-8")
    return re.findall(r'add_argument\("([a-z_]+)"', text)


# --- output auditor -------------------------------------------------------------------------
# An earlier version of this test BLACKLISTED identifier names (row, clip, audio, text, exc...).
# A review mutation-tested it and 8 of 10 realistic leaks passed: `rows[0]` defeats `\brow\b`,
# aliasing a path to `a` defeats everything, and `logging.warning(...)` is not a `print` at all.
# A blacklist can only encode the last bug. This is an ALLOWLIST: it states what may reach the
# log -- counts, the run id, voter names, exception *types* -- and flags everything else.
SAFE_NAMES = {"run_id", "name", "done", "empty", "blank", "have", "seconds", "before",
              # `entry`, `summary` and `result` hold counts-only dicts; their key sets are
              # asserted separately by test_returned_and_logged_dicts_carry_only_counts.
              "entry", "summary", "result",
              # `detail` is the ONE exception value allowed to reach the log, and only as
              # `str(exc) if isinstance(exc, ModelLoadError) else type(exc).__name__`. A
              # ModelLoadError is raised only where no audio has been opened, so its message can
              # name a model path or a missing CUDA library but never a clip.
              # test_detail_is_bound_only_to_a_model_load_message pins the expression, and
              # test_model_load_error_is_raised_only_at_model_load pins where it can be raised.
              "detail"}
SAFE_KEYS = {"voter", "segments", "empty", "seconds", "run", "voters", "resumed"}
# Only these yield a scalar REGARDLESS of their argument. str()/int()/round() do not:
# str(exc) renders whatever exc is, so those descend into their arguments.
SCALAR_CALLS = {"len", "type"}
DESCEND_CALLS = {"str", "int", "round", "float", "sorted", "sum"}
OUTPUT_ATTRS = {"write", "writelines", "debug", "info", "warning", "error", "critical",
                "exception", "log", "warn", "print", "print_exc", "format_exc",
                "print_exception"}


def _audit_expr(node):
    bad = []

    bound = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in n.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name):
                        bound.add(sub.id)

    class V(ast.NodeVisitor):
        def visit_Call(self, n):
            f = n.func
            if isinstance(f, ast.Name) and f.id in SCALAR_CALLS:
                return                                    # len()/type() yield scalars
            if isinstance(f, ast.Name) and f.id in DESCEND_CALLS:
                for a in n.args:
                    self.visit(a)                         # value depends on the argument
                return
            if (isinstance(f, ast.Attribute) and f.attr == "dumps"
                    and isinstance(f.value, ast.Name) and f.value.id == "json"):
                for a in n.args:
                    self.visit(a)                         # audit what is being serialised
                return
            bad.append("call:" + ast.unparse(n))

        def visit_Attribute(self, n):
            if n.attr == "__name__":
                return                                    # type(exc).__name__
            bad.append("attr:" + ast.unparse(n))

        def visit_Subscript(self, n):
            key = n.slice
            # A safe KEY is not enough: `rows[0]["voter"]` would sail through. The object being
            # subscripted must itself be allowlisted or comprehension-bound.
            base_ok = isinstance(n.value, ast.Name) and n.value.id in SAFE_NAMES | bound
            if base_ok and isinstance(key, ast.Constant) and key.value in SAFE_KEYS:
                return
            bad.append("subscript:" + ast.unparse(n))

        def visit_Name(self, n):
            if n.id not in SAFE_NAMES | bound:
                bad.append("name:" + n.id)

    V().visit(node)
    return bad


def audit(source):
    """Return a list of (kind, code) findings: output calls that may carry data."""
    tree = ast.parse(source)
    findings = []
    # Any route that gets `print` under another name defeats a call-site check. Rather than
    # enumerate the routes (p = print, p: object = print, p, q = print, print, from builtins
    # import print as p, functools.partial(print), getattr(builtins, "print")), refuse the
    # *value* `print` anywhere it is bound or passed. The script never does this.
    def _mentions_print(v):
        return any(isinstance(n, ast.Name) and n.id == "print" for n in ast.walk(v))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            value = getattr(node, "value", None)
            if value is not None and _mentions_print(value):
                findings.append(("print-alias", ast.unparse(node)))
        elif isinstance(node, (ast.For, ast.AsyncFor)) and _mentions_print(node.iter):
            findings.append(("print-alias", ast.unparse(node.iter)))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "print":
                    findings.append(("print-alias", f"from {node.module} import print as {a.asname}"))
        elif isinstance(node, ast.Call):
            f = node.func
            # `print` handed to a wrapper: functools.partial(print), map(print, ...), etc.
            if any(isinstance(a, ast.Name) and a.id == "print" for a in node.args):
                findings.append(("print-passed-as-value", ast.unparse(node)))
            if isinstance(f, ast.Name) and f.id == "getattr" and node.args \
                    and any(isinstance(a, ast.Constant) and a.value == "print" for a in node.args):
                findings.append(("print-via-getattr", ast.unparse(node)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        is_print = (isinstance(f, ast.Name) and f.id == "print") or \
                   (isinstance(f, ast.Attribute) and f.attr == "print")
        is_other_output = False
        if isinstance(f, ast.Attribute) and f.attr in OUTPUT_ATTRS and not is_print:
            # `.write` on a file handle is the run's own output to the volume, not the log.
            # Only the process streams reach Modal's log.
            if f.attr in {"write", "writelines"}:
                recv = ast.unparse(f.value)
                # Match the whole family rather than three spellings: sys.__stdout__,
                # sys.stdout.buffer, os.write all reach the same place.
                is_other_output = (recv.startswith(("sys.", "os."))
                                   or recv in {"sys", "os"}
                                   or "stdout" in recv or "stderr" in recv)
            else:
                is_other_output = True
        if is_other_output:
            findings.append(("non-print-output", ast.unparse(node)))
            continue
        if not is_print:
            continue
        for arg in node.args:
            for b in _audit_expr(arg):
                findings.append(("print-arg", f"{ast.unparse(node)}  [{b}]"))
        # sep= and end= reach the log exactly like a positional argument does.
        for kw in node.keywords:
            if kw.arg in {"sep", "end"}:
                for b in _audit_expr(kw.value):
                    findings.append(("print-kwarg", f"{ast.unparse(node)}  [{kw.arg}: {b}]"))
    return findings


# The allowlist is a PROMISE that these names hold only counts, ids and voter names. A promise
# has to be checked: `done = rows[0]["audio"]` followed by `print(f"{done}")` would otherwise
# sail through, because `done` is allowlisted. So every binding of an allowlisted name is audited
# too, and only these shapes are accepted.
DETAIL_EXPR = ("str(exc) if isinstance(exc, ModelLoadError) "
               "else type(exc).__name__")
SAFE_BINDING_NAMES = {"wanted"}          # voter ids, derived from VOTERS / the --voters string
SAFE_BINDING_CALLS = {"len", "round", "int", "float", "sorted"}


def audit_bindings(source):
    """Findings for any allowlisted name bound to something that is not provably a count,
    a literal, a boolean or a checked dict."""
    tree = ast.parse(source)
    findings = []

    def safe(v):
        if isinstance(v, ast.Constant):
            return True
        if isinstance(v, (ast.Tuple, ast.List, ast.Set)):
            return all(safe(e) for e in v.elts)
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not):
            return True                                   # `not x` is a bool whatever x is
        if isinstance(v, ast.Compare):
            return True                                   # comparisons are bools
        if isinstance(v, ast.Name):
            return v.id in SAFE_BINDING_NAMES
        if isinstance(v, ast.Dict):
            return True                                   # keys checked by the dict test
        if isinstance(v, ast.BinOp):
            return safe(v.left) and safe(v.right)
        if isinstance(v, ast.BoolOp):
            return all(safe(e) for e in v.values)
        if isinstance(v, ast.Call):
            f = v.func
            if isinstance(f, ast.Name) and f.id in SAFE_BINDING_CALLS:
                return True
            # decode.remote(...) returns the counts-only dict the dict test checks.
            if isinstance(f, ast.Attribute) and f.attr == "remote":
                return True
        return False

    for n in ast.walk(tree):
        # Binding forms that take a value we can judge.
        targets, value = [], None
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            targets, value = [n.target], getattr(n, "value", None)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            targets, value = [n.target], n.iter
        if value is not None:
            for tgt in targets:
                for sub in ast.walk(tgt):
                    if not (isinstance(sub, ast.Name) and sub.id in SAFE_NAMES):
                        continue
                    # One exemption, pinned to an exact expression rather than a shape:
                    # `detail` may hold a ModelLoadError's message, which can name a model path
                    # or a CUDA library but never a clip. Any other binding of `detail`, and any
                    # drift in this expression, is a finding.
                    if sub.id == "detail" and ast.unparse(value) == DETAIL_EXPR:
                        continue
                    if not safe(value):
                        findings.append((sub.id, ast.unparse(value)))
        # Binding forms with no judgeable value: an allowlisted name bound here is always a
        # laundering route. `except Exception as done:` then `print(f"{done}")` is the sharpest
        # -- it puts an exception VALUE behind a name both auditors trust.
        if isinstance(n, ast.ExceptHandler) and n.name in SAFE_NAMES:
            findings.append((n.name, f"except ... as {n.name} (exception value)"))
        if isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    for sub in ast.walk(item.optional_vars):
                        if isinstance(sub, ast.Name) and sub.id in SAFE_NAMES:
                            findings.append((sub.id, f"with ... as {sub.id}"))
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound = a.asname or a.name.split(".")[0]
                if bound in SAFE_NAMES:
                    findings.append((bound, f"import ... as {bound}"))
    return findings


# Known leak shapes. test_the_auditor_catches_known_leaks keeps this honest: if the auditor is
# ever weakened, these fail before the real file passes for the wrong reason.
KNOWN_LEAKS = [
    ('print(f"{rows[0]}")', "a whole manifest row"),
    ('a = rows[0]["audio"]\nprint(f"{a}")', "a clip path aliased to one letter"),
    ('print(f"{hyp}")', "a transcript under another name"),
    ('print(f"{rows}")', "the entire row list"),
    ('print(f"{root}")', "the dataset path"),
    ('print(f"{target}")', "the output path"),
    ('logging.warning(rows[0]["audio"])', "output that is not a print"),
    ('sys.stderr.write(str(rows[0]))', "a direct stderr write"),
    ('builtins.print(rows[0]["audio"])', "print reached through an attribute"),
    ("print(f\"{row['audio']}\")", "a clip path, the original defect"),
    ("print(str(exc))", "an exception value rather than its type"),
    ('print(f"{e}")', "an exception aliased to e"),
    ('print(f"{text}")', "transcript text"),
    ('print(f"{segments[0].text}")', "text off a segment object"),
    ("print(str(row))", "str() of a row"),
    ("print(f\"{[r['audio'] for r in rows]}\")", "clip paths through a comprehension"),
]


def test_the_auditor_catches_known_leaks():
    """The auditor is only worth running if it fails on things that should fail."""
    missed = [why for src, why in KNOWN_LEAKS if not audit(src)]
    assert not missed, "auditor no longer catches: " + "; ".join(missed)


def test_the_auditor_accepts_the_real_prints():
    """And only worth trusting if it does not cry wolf on the lines the script really has."""
    for src in ('print(f"SEGMENTS {len(rows)}", flush=True)',
                'print(f"FAILED {run_id}: {type(exc).__name__}", flush=True)',
                'print("VOTER " + json.dumps(entry), flush=True)',
                'print(f"PROGRESS {name} {done}/{len(rows)}", flush=True)',
                r'sink.write(json.dumps({"id": row["id"], "text": text}) + "\n")'):
        assert not audit(src), f"false positive on: {src}"


def test_nothing_in_modal_decode_can_reach_the_log_with_competition_data():
    findings = audit(SOURCE)
    assert not findings, "output may carry competition data:\n" + "\n".join(
        f"  [{kind}] {code}" for kind, code in findings)


def test_returned_and_logged_dicts_carry_only_counts():
    """`summary`, `entry` and the returned dict are allowlisted by name in the auditor, so their
    key sets have to be checked here or the allowlist is a hole."""
    allowed = {"run", "dataset", "segments", "voters", "voter", "empty", "seconds",
               "resumed", "guard"}
    tree = ast.parse(SOURCE)
    checked = 0
    for fn_name in ("decode", "_decode"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        # Only what LEAVES the function matters: values returned to the caller (and printed by
        # the entrypoint) or passed to print. `rows` holds {"id","audio"} internally and never
        # reaches a log, so walking every Dict in the body would be checking the wrong thing.
        leaving = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value]
        leaving += [a for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "print" for a in n.args]
        for root in leaving:
            for node in ast.walk(root):
                if isinstance(node, ast.Dict) and node.keys:
                    keys = {k.value for k in node.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                    bad = keys - allowed
                    assert not bad, f"{fn_name} returns/logs non-count keys: {sorted(bad)}"
                    checked += 1
    assert checked, "found nothing to check; the test would pass vacuously"


def test_library_logging_is_silenced():
    """faster-whisper prints two lines per clip once anything calls logging.basicConfig; this
    repo already learned that in submission_src/main.py:102-123."""
    tree = ast.parse(SOURCE)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_silence_libraries"), None)
    assert fn is not None, "no _silence_libraries(); library output would reach the Modal log"
    body = ast.unparse(fn)
    assert "logging.disable" in body
    assert "faster_whisper" in body and "ctranslate2" in body
    decode_fn = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "decode")
    assert "_silence_libraries()" in ast.unparse(decode_fn), "silencing is never called"


def test_dataset_is_allowlisted_so_a_typo_cannot_decode_dev():
    assert "dev_folds" not in modal_decode.ALLOWED_DATASETS
    assert "bangor_g7" in modal_decode.ALLOWED_DATASETS
    decode_fn = next(n for n in ast.walk(ast.parse(SOURCE))
                     if isinstance(n, ast.FunctionDef) and n.name == "decode")
    assert "ALLOWED_DATASETS" in ast.unparse(decode_fn), "allowlist is never enforced"


def test_the_guard_catches_baseexception():
    """Modal delivers timeout and preemption as KeyboardInterrupt/SystemExit, which
    `except Exception` does not catch -- and those unwind through frames holding a clip path."""
    decode_fn = next(n for n in ast.walk(ast.parse(SOURCE))
                     if isinstance(n, ast.FunctionDef) and n.name == "decode")
    handlers = [h for n in ast.walk(decode_fn) if isinstance(n, ast.Try) for h in n.handlers]
    caught = {ast.unparse(h.type) for h in handlers if h.type is not None}
    assert "BaseException" in caught, f"only catches {caught}"


def test_nothing_runs_before_the_try():
    """v2 claimed 'decode() catches every exception' while mkdir, the guard read and the ATTEMPT
    write sat outside the try. The claim has to be true as written."""
    decode_fn = next(n for n in ast.walk(ast.parse(SOURCE))
                     if isinstance(n, ast.FunctionDef) and n.name == "decode")
    tries = [s for s in decode_fn.body if isinstance(s, ast.Try)]
    assert len(tries) == 1, f"expected exactly one top-level try, found {len(tries)}"
    # A `try: ... finally: pass` with no handler still lets the exception escape, so a nested
    # try before the real one would be a bypass. Require the one try to actually handle.
    assert tries[0].handlers, "the top-level try has no except handler"
    for stmt in decode_fn.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if stmt is tries[0]:
            continue
        # Only the docstring. Skipping bare `ast.Expr` would skip every CALL statement too --
        # `out.mkdir(...)` before the try is an ast.Expr, and that is precisely the v2 defect
        # this test exists to prevent returning.
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                and isinstance(stmt.value.value, str):
            continue
        raise AssertionError(f"statement outside the try: {ast.unparse(stmt)}")


def test_decode_body_is_wrapped_so_tracebacks_cannot_escape():
    tree = ast.parse(SOURCE)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decode")
    handlers = [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers]
    assert handlers, "decode() has no except block; a traceback would name a clip"
    raises = [n for h in handlers for n in ast.walk(h) if isinstance(n, ast.Raise)]
    # `raise X from None` is Raise(cause=Constant(None)); a bare `raise X` is Raise(cause=None),
    # which would keep the original traceback and with it the clip path.
    assert raises, "except block does not re-raise"
    assert all(isinstance(r.cause, ast.Constant) and r.cause.value is None for r in raises), \
        "re-raise must use `from None` so the original chain is dropped"


def test_a_finished_run_is_guarded():
    fn = next(n for n in ast.walk(ast.parse(SOURCE))
              if isinstance(n, ast.FunctionDef) and n.name == "decode")
    assert "GUARD" in ast.unparse(fn), "no rerun guard: a relaunch would redecode at full price"


KNOWN_REBINDINGS = [
    ('done = rows[0]["audio"]', "a clip path bound to the allowlisted `done`"),
    ('name = row["transcript"]', "transcript text bound to the allowlisted `name`"),
    ('seconds = rows', "the row list bound to the allowlisted `seconds`"),
    ('for name in rows:\n    pass', "loop binding an allowlisted name to rows"),
]


def test_the_binding_auditor_catches_rebinding_attacks():
    """An allowlist is only as good as the promise it makes about its names."""
    missed = [why for src, why in KNOWN_REBINDINGS if not audit_bindings(src)]
    assert not missed, "binding auditor no longer catches: " + "; ".join(missed)


def test_no_allowlisted_name_is_bound_to_anything_unsafe():
    findings = audit_bindings(SOURCE)
    assert not findings, "allowlisted names bound to unproven values:\n" + "\n".join(
        f"  {n} <- {v}" for n, v in findings)


# --- behavioural tests ----------------------------------------------------------------------
# Every other test in this file asserts what the source TEXT says. That is how 250 green tests
# coexisted with a script Modal could not launch: a patch inserted a helper between
# `@app.function(...)` and `def decode`, so the decorator bound to the helper. `decode` became a
# plain function with no `.remote`, and the A100/timeout/retries settings were attached to
# something that does no decoding. These tests ask what the module IS.


def test_decode_is_the_modal_function():
    import modal

    assert isinstance(modal_decode.decode, modal.functions.Function), (
        f"decode is {type(modal_decode.decode).__name__}; the @app.function decorator is on "
        "the wrong object and `modal run ...::decode` will resolve to nothing")
    assert hasattr(modal_decode.decode, "remote"), "decode has no .remote"
    assert list(modal_decode.app.registered_functions) == ["decode"]


def test_silence_libraries_is_a_plain_callable():
    """It must be called inside the container, so it must not itself be a Modal Function."""
    assert callable(modal_decode._silence_libraries), (
        f"_silence_libraries is {type(modal_decode._silence_libraries).__name__}; calling it "
        "would raise TypeError inside decode()")
    modal_decode._silence_libraries()          # must not raise


def test_modal_resolves_the_reference_the_watchdog_builds():
    """`modal_watchdog.watch()` builds `modal run --detach <script>::<stage>`. Resolve that exact
    reference through Modal's own resolver rather than trusting that it looks right."""
    import modal.cli.import_refs as import_refs

    ref = import_refs.parse_import_ref("scripts/modal_decode.py::decode")
    runnable, _ = import_refs.import_and_filter(
        ref, base_cmd="modal run", accept_local_entrypoint=True, accept_webhook=False)
    assert runnable is not None, "modal run scripts/modal_decode.py::decode resolves to nothing"


def test_the_container_settings_are_on_decode():
    """The $14.11 per-attempt ceiling and the A100 rate are only real if the decorator sits on
    the function that decodes.

    `_FunctionSpec` exposes gpus/cpu/memory/volumes/image but NOT timeout or startup_timeout,
    so those two are checked where they are visible: in the decorator attached to `def decode`.
    Checked together, the pair pins the decorator to the right object.
    """
    spec = modal_decode.decode.spec
    assert spec.gpus == modal_decode.GPU, spec.gpus
    assert spec.cpu == modal_decode.CPU, spec.cpu
    assert spec.memory == modal_decode.MEMORY, spec.memory
    assert modal_decode.VOLUME in {str(k) for k in spec.volumes}, spec.volumes

    tree = ast.parse(SOURCE)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decode")
    decorators = " ".join(ast.unparse(d) for d in fn.decorator_list)
    assert "app.function" in decorators, "decode carries no @app.function decorator"
    for setting in ("timeout=240 * 60", "startup_timeout=STARTUP_TIMEOUT", "retries=0"):
        assert setting in decorators, f"{setting} is not on decode: {decorators}"


# `run_id` is the only allowlisted name that legitimately appears as a parameter; it is a run
# identifier the operator chose. Any other allowlisted name used as a parameter would let a
# caller bind a clip path to a name the auditor trusts.
SAFE_PARAM_NAMES = {"run_id"}


def test_no_function_parameter_shadows_an_allowlisted_name():
    tree = ast.parse(SOURCE)
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if a.arg in SAFE_NAMES and a.arg not in SAFE_PARAM_NAMES:
                offenders.append(f"{fn.name}({a.arg})")
    assert not offenders, ("allowlisted names used as parameters, so a caller could bind them "
                           "to anything: " + ", ".join(offenders))


# Channel and laundering attacks. Each was OPEN at some point and was closed in response to a
# review; without a regression test a later edit could silently reopen any of them, which is the
# durability argument the falsification artifact makes for itself.
KNOWN_CHANNEL_ATTACKS = [
    ('print("x", sep=rows[0]["audio"])', "clip path through print's sep="),
    ('print("x", end=row["transcript"])', "transcript through print's end="),
    ('sys.stdout.writelines([rows[0]["audio"]])', "writelines instead of write"),
    ('sys.__stdout__.write(str(rows[0]))', "sys.__stdout__ instead of sys.stdout"),
    ('sys.stdout.buffer.write(b"x")', "the buffer beneath sys.stdout"),
    ('traceback.print_exc()', "a traceback printed wholesale"),
    ('print(traceback.format_exc())', "a traceback formatted then printed"),
    ('p = print\np(rows[0]["audio"])', "print aliased by assignment"),
    ('p: object = print\np(rows[0]["audio"])', "print aliased by annotated assignment"),
    ('p, q = print, print\np(rows[0]["audio"])', "print aliased by tuple unpacking"),
    ('from builtins import print as p\np(rows[0]["audio"])', "print aliased at import"),
    ('f = functools.partial(print)\nf(rows[0]["audio"])', "print wrapped in functools.partial"),
    ('functools.partial(print)(rows[0]["audio"])', "partial(print) called inline"),
    ('getattr(builtins, "print")(rows[0]["audio"])', "print reached through getattr"),
    ('print(f\'{rows[0]["voter"]}\')', "a safe key subscripted on an unsafe object"),
    ('try:\n    pass\nexcept Exception as done:\n    print(f"{done}")',
     "an exception value laundered through an allowlisted name"),
    ('with open("x") as done:\n    print(f"{done}")',
     "a file handle laundered through an allowlisted name"),
    # These four branches existed but nothing exercised them: a review deleted each and the
    # suite stayed green. By this file's own rule that made them promises, not properties.
    ('for p in [print]:\n    p(rows[0]["audio"])', "print aliased by a for target"),
    ('(p := print)(rows[0]["audio"])', "print aliased by a walrus"),
    ('p = None\np += print\np(rows[0]["audio"])', "print aliased by augmented assignment"),
    ('import json as done\nprint(f"{done}")', "an import bound to an allowlisted name"),
]

REAL_OUTPUT_LINES = [
    'print(f"SEGMENTS {len(rows)}", flush=True)',
    'print(f"FAILED {run_id}: {type(exc).__name__}", flush=True)',
    'print("VOTER " + json.dumps(entry), flush=True)',
    'print(f"PROGRESS {name} {done}/{len(rows)}", flush=True)',
    'print(f"GUARD {run_id}: summary.json exists; not decoding", flush=True)',
    'print(f"IDFILTER kept {len(rows)} of {before}", flush=True)',
    'print(f"RESUME {name}: {have} already decoded; skipping", flush=True)',
    'print(f"RESUME {name}: {have}/{len(rows)} incomplete; decoding again", flush=True)',
    'print("REPORT " + json.dumps({"run": run_id, "segments": len(rows), '
    '"voters": [v["voter"] for v in summary]}), flush=True)',
    'print(json.dumps(result))',
    r'sink.write(json.dumps({"id": row["id"], "text": text}) + "\n")',
    r'marker.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")',
]


def _audit_all(source):
    return audit(source) + audit_bindings(source)


def test_channel_and_laundering_attacks_are_all_closed():
    missed = [why for src, why in KNOWN_CHANNEL_ATTACKS if not _audit_all(src)]
    assert not missed, "reopened: " + "; ".join(missed)


def test_every_real_output_line_is_accepted():
    """Eleven lines, not six: every print and every .write in the module. An auditor that
    flagged a real line would be quietly disabled the first time it cried wolf."""
    flagged = [s for s in REAL_OUTPUT_LINES if _audit_all(s)]
    assert not flagged, "false positives: " + "; ".join(flagged)


def test_the_module_has_no_output_call_outside_the_checked_set():
    """Guards the list above from drifting out of date as the script changes."""
    tree = ast.parse(SOURCE)
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "print") or \
                    (isinstance(f, ast.Attribute) and f.attr in {"print", "write", "writelines"}):
                calls += 1
    assert calls == len(REAL_OUTPUT_LINES), (
        f"module has {calls} output calls but REAL_OUTPUT_LINES lists {len(REAL_OUTPUT_LINES)}; "
        "update the list and re-audit")


def test_detail_is_bound_only_to_a_model_load_message():
    """`detail` is allowlisted, so its binding is the hole. Pin the exact expression."""
    tree = ast.parse(SOURCE)
    bindings = [ast.unparse(n.value) for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                for tgt in n.targets
                for sub in ast.walk(tgt)
                if isinstance(sub, ast.Name) and sub.id == "detail"]
    assert bindings == [DETAIL_EXPR], bindings


def test_model_load_error_is_raised_only_at_model_load():
    """The exemption holds only while ModelLoadError cannot escape from the decode loop, where
    an exception CAN name a clip."""
    tree = ast.parse(SOURCE)
    raises = [n for n in ast.walk(tree)
              if isinstance(n, ast.Raise) and n.exc is not None
              and "ModelLoadError" in ast.unparse(n.exc)]
    assert len(raises) == 1, f"ModelLoadError raised {len(raises)} times; expected exactly 1"
    # It must sit in a handler whose try contains the WhisperModel construction and nothing that
    # touches a row.
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(r in ast.walk(node) for r in raises):
            body = ast.unparse(node.body)
            assert "WhisperModel(" in body, body
            assert "transcribe" not in body and "row" not in body, body
            return
    raise AssertionError("the ModelLoadError raise is not inside a try around WhisperModel")


# --- surface lock ---------------------------------------------------------------------------
# Six reviews chased attack SHAPES and a seventh still found nine more: `**kwargs` smuggling,
# `fh = sys.stdout`, `_o = sys.stdout.write`, `open("/proc/self/fd/1","w")`,
# `__builtins__["print"]`, `os.system`, `subprocess.run`. Enumerating attacks is unbounded.
# This module is small and its legitimate surface is fully known -- eight imports, no bare
# `open(`, eleven output calls -- so lock the surface instead. Anything outside it is a finding,
# whether or not anyone has thought of the attack that uses it.

ALLOWED_IMPORTS = {
    "__future__", "modal", "logging", "ctranslate2", "time", "pathlib", "json", "faster_whisper",
}
# Modules that give a process another way to emit bytes, or to reach `print` indirectly.
FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "builtins", "functools", "traceback", "warnings", "shutil",
    "socket", "urllib", "requests", "http", "ctypes", "multiprocessing", "threading",
}


def audit_surface(source):
    """Findings for anything outside this module's known-good surface."""
    tree = ast.parse(source)
    findings = []
    for n in ast.walk(tree):
        # 1. Imports: allowlist, with an explicit forbidden set for the clearest offenders.
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""])
            for m in mods:
                root = m.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    findings.append(("forbidden-import", ast.unparse(n)))
                elif root not in ALLOWED_IMPORTS:
                    findings.append(("unknown-import", ast.unparse(n)))
        # 2. Bare open(): every file this module touches goes through a Path, so a bare
        #    open("/proc/self/fd/1", "w") has no legitimate reason to appear.
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open":
            findings.append(("bare-open", ast.unparse(n)))
        # 3. `**` in a print call: the keyword's `arg` is None, so no per-keyword check can see
        #    inside it. print("SEG", **kw) with a clip path in kw passed every earlier test.
        if isinstance(n, ast.Call):
            f = n.func
            is_print = (isinstance(f, ast.Name) and f.id == "print") or \
                       (isinstance(f, ast.Attribute) and f.attr == "print")
            if is_print and any(kw.arg is None for kw in n.keywords):
                findings.append(("print-double-star", ast.unparse(n)))
        # 4. Binding a process stream, or anything's `.write`, to a name.
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            v = getattr(n, "value", None)
            if v is not None:
                rendered = ast.unparse(v)
                if any(s in rendered for s in ("sys.stdout", "sys.stderr", "sys.__stdout__",
                                               "sys.__stderr__")):
                    findings.append(("stream-alias", ast.unparse(n)))
                if re.search(r"\.write(lines)?\b(?!\s*\()", rendered):
                    findings.append(("write-method-alias", ast.unparse(n)))
        # 5. Reaching builtins through a subscript.
        if isinstance(n, ast.Subscript):
            base = ast.unparse(n.value)
            if base in {"__builtins__", "builtins", "globals()", "vars()"}:
                findings.append(("builtins-subscript", ast.unparse(n)))
    return findings


KNOWN_SURFACE_ATTACKS = [
    ('print(**{"sep": rows[0]["audio"]})', "a clip path through print's **kwargs"),
    ('kw = {"end": row["transcript"]}\nprint("SEG", **kw)', "**kwargs built in a variable"),
    ('print(**dict(sep=rows[0]["audio"]))', "**dict(...)"),
    ('fh = sys.stdout\nfh.write(rows[0]["audio"])', "stdout bound to a name"),
    ('_o = sys.stdout.write\n_o(rows[0]["audio"])', "the write method itself bound"),
    ('open("/proc/self/fd/1", "w").write(rows[0]["audio"])', "a file descriptor path"),
    ('__builtins__["print"](rows[0]["audio"])', "print through a builtins subscript"),
    ('import os\nos.system("echo " + rows[0]["audio"])', "os.system"),
    ('import subprocess\nsubprocess.run(["echo", rows[0]["audio"]])', "subprocess"),
    ('import functools\nf = functools.partial(print)', "importing functools at all"),
]


def test_the_surface_lock_catches_what_shape_auditing_missed():
    """These nine were all OPEN under shape auditing and were proved exploitable: two of them
    left the full suite green with a real clip path reaching the log."""
    missed = [why for src, why in KNOWN_SURFACE_ATTACKS if not audit_surface(src)]
    assert not missed, "surface lock does not catch: " + "; ".join(missed)


def test_the_module_stays_inside_its_known_surface():
    findings = audit_surface(SOURCE)
    assert not findings, "outside the known surface:\n" + "\n".join(
        f"  [{k}] {c}" for k, c in findings)
