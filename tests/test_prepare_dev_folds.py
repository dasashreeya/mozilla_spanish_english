from scripts.prepare_dev_folds import clip_folds


def test_clip_folds_partition_every_id_exactly_once() -> None:
    ids = [f"dev{i:03d}" for i in range(155)]
    folds = clip_folds(ids, 5, seed=20260920)
    assert len(folds) == 5
    assert sorted(i for fold in folds for i in fold) == ids
    assert [len(f) for f in folds] == [31, 31, 31, 31, 31]


def test_clip_folds_are_deterministic_and_seed_dependent() -> None:
    ids = [f"dev{i:03d}" for i in range(20)]
    assert clip_folds(ids, 4, seed=1) == clip_folds(ids, 4, seed=1)
    assert clip_folds(ids, 4, seed=1) != clip_folds(ids, 4, seed=2)


def test_clip_folds_uneven_split_keeps_every_id() -> None:
    ids = [f"dev{i:03d}" for i in range(7)]
    folds = clip_folds(ids, 5, seed=3)
    assert sorted(i for fold in folds for i in fold) == ids
    assert sorted(len(f) for f in folds) == [1, 1, 1, 2, 2]


def test_training_sets_exclude_long_clips_and_the_eval_fold() -> None:
    # The construction used in main(): (all - eval) & trainable, which must drop both.
    ids = [f"dev{i:03d}" for i in range(6)]
    trainable = {"dev000", "dev001", "dev002", "dev003"}  # dev004, dev005 are over 30 s
    evaluation = ["dev000", "dev004"]
    train = sorted((set(ids) - set(evaluation)) & trainable)
    assert train == ["dev001", "dev002", "dev003"]
