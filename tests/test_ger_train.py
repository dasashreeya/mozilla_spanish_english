"""Guards for the GER LoRA trainer, written after review found two collator defects.

The trainer runs on Modal, so nothing here touches a GPU. What it checks is the part that
review proved broken before it ever ran: the collator. Both defects were in three lines.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _collator_class():
    """Load PromptMaskedCollator without importing the module.

    scripts/modal_ger_train.py must stay self-contained -- Modal ships only the entrypoint file,
    and a cross-file import resolves locally then dies in the container (that is how ger-probe1
    failed). So the class cannot be moved to a shared module, and the module cannot be imported
    in an environment without `modal`. Extracting the class definition from the source and
    exec'ing it tests the real source text, which is what we want anyway.
    """
    import ast

    src = Path(ROOT / "scripts" / "modal_ger_train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "PromptMaskedCollator")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<collator>", "exec"), namespace)
    return namespace["PromptMaskedCollator"]


def _tokenizer():
    """A real fast tokenizer, built offline so the test needs no network."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5}
    inner = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[PAD]"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=inner, pad_token="[PAD]", eos_token="[EOS]")


def test_collator_pads_labels_and_keeps_the_prompt_mask():
    """The defect: DataCollatorForLanguageModeling cannot pad `labels`, and overwrites them with
    a clone of input_ids -- discarding the mask the trainer builds."""
    torch = pytest.importorskip("torch")
    PromptMaskedCollator = _collator_class()

    tok = _tokenizer()
    # Two examples of DIFFERENT length, prompt-masked exactly as build() does.
    features = [
        {"input_ids": [2, 3, 4], "attention_mask": [1, 1, 1], "labels": [-100, 3, 4]},
        {"input_ids": [2, 3, 4, 5, 1], "attention_mask": [1, 1, 1, 1, 1],
         "labels": [-100, -100, 4, 5, 1]},
    ]
    batch = PromptMaskedCollator(tok)(features)

    assert batch["labels"].shape == batch["input_ids"].shape
    # The mask survived: position 0 of row 0 and positions 0-1 of row 1 are still ignored.
    assert batch["labels"][0][0].item() == -100
    assert batch["labels"][1][:2].tolist() == [-100, -100]
    # Padding is -100, never the pad token id, and never a copy of input_ids.
    assert batch["labels"][0][-1].item() == -100, "short row not padded with -100"
    assert not torch.equal(batch["labels"], batch["input_ids"]), \
        "labels are a clone of input_ids; the prompt mask was discarded"
    # The real EOS (id 1) is still a supervised target, not masked away.
    assert 1 in batch["labels"][1].tolist(), "EOS was masked; the model would never learn to stop"


def test_collator_respects_left_padding():
    pytest.importorskip("torch")
    PromptMaskedCollator = _collator_class()

    tok = _tokenizer()
    tok.padding_side = "left"
    features = [
        {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [-100, 3]},
        {"input_ids": [2, 3, 4, 5], "attention_mask": [1, 1, 1, 1], "labels": [-100, 3, 4, 5]},
    ]
    batch = PromptMaskedCollator(tok)(features)
    assert batch["labels"].shape == batch["input_ids"].shape
    # Left padding must put the -100s at the FRONT so labels line up with input_ids.
    assert batch["labels"][0][:2].tolist() == [-100, -100]
    assert batch["labels"][0][-1].item() == 3


def test_the_broken_collator_is_not_used():
    """Checked on the AST, not the text: the module's own docstring explains why that collator
    is wrong and would match a text search."""
    import ast

    tree = ast.parse(Path(ROOT / "scripts" / "modal_ger_train.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "transformers":
            names = {a.name for a in node.names}
            assert "DataCollatorForLanguageModeling" not in names, names
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            assert name != "DataCollatorForLanguageModeling", ast.unparse(node)


def test_bases_are_open_data_only():
    """The Bangor GPL clause permits LLM training only if all the model's training data is
    public, so open-weights-only models must not be reachable."""
    import ast

    src = Path(ROOT / "scripts" / "modal_ger_train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assign = next(n for n in tree.body
                  if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == "BASES" for t in n.targets))
    bases = ast.literal_eval(assign.value)
    assert set(bases) == {"smollm3", "olmo2"}
    for repo in bases.values():
        assert repo.split("/")[0] in {"HuggingFaceTB", "allenai"}, repo
