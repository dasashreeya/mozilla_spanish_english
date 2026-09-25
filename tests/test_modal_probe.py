"""The probe duplicates constants from modal_decode on purpose; keep them identical.

ger-probe1 failed in the container with ModuleNotFoundError because modal_probe imported from
modal_decode and Modal ships only the entrypoint file. Duplication is the fix, and duplication
rots, so this test is the thing that stops it rotting.
"""
import modal
from pathlib import Path

from scripts import modal_decode, modal_probe


def test_probe_is_a_modal_function_that_resolves():
    import modal.cli.import_refs as import_refs

    assert isinstance(modal_probe.probe, modal.functions.Function)
    assert list(modal_probe.app.registered_functions) == ["probe"]
    ref = import_refs.parse_import_ref("scripts/modal_probe.py::probe")
    runnable, _ = import_refs.import_and_filter(
        ref, base_cmd="modal run", accept_local_entrypoint=True, accept_webhook=False)
    assert runnable is not None


def test_probe_imports_nothing_from_the_decode_module():
    """A cross-file import resolves locally and dies in the container.

    Checked on the AST, not the text: the module's own comments explain this failure and
    contain the very strings a text search would match.
    """
    import ast

    tree = ast.parse(Path(modal_probe.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("modal_decode"), ast.unparse(node)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("modal_decode"), ast.unparse(node)


def test_duplicated_constants_match_modal_decode():
    for name in ("GPU", "CPU", "MEMORY", "VOLUME", "STARTUP_TIMEOUT"):
        assert getattr(modal_probe, name) == getattr(modal_decode, name), name
    assert modal_probe.VOTERS["turbo"] == modal_decode.VOTERS["turbo"]


def test_probe_reads_no_dataset():
    """The whole justification for printing tracebacks in full.

    Comments and docstrings are stripped first -- the module explains at length WHY it reads no
    dataset, and a text search would flag that explanation.
    """
    import ast

    tree = ast.parse(Path(modal_probe.__file__).read_text(encoding="utf-8"))
    # ast.unparse drops comments; strip docstrings too.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    for forbidden in ("manifest", "bangor", "VOLUME /", "/vol/", "row["):
        assert forbidden not in code, f"probe code references {forbidden}; it must touch no data"
    # It may name the volume constant (the decorator mounts it) but must never build a path
    # into a dataset directory.
    assert "Path(VOLUME)" not in code
