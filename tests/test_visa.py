"""VisA indexing. These tests use a small CSV, not the 1.8 GiB archive."""
from __future__ import annotations

from pathlib import Path

import pytest

from parallax.visa import (
    ALL_CLASSES,
    DEFORMABLE_CLASSES,
    RIGID_CLASSES,
    Sample,
    load_split,
    split_csv_path,
    verify,
)

CSV_HEADER = "object,split,label,image,mask\n"
CSV_ROWS = (
    "candle,train,normal,candle/Data/Images/Normal/0836.JPG,\n"
    "pcb1,test,anomaly,pcb1/Data/Images/Anomaly/000.JPG,pcb1/Data/Masks/Anomaly/000.png\n"
    "cashew,test,normal,cashew/Data/Images/Normal/001.JPG,\n"
    "macaroni1,test,anomaly,macaroni1/Data/Images/Anomaly/002.JPG,macaroni1/Data/Masks/Anomaly/002.png\n"
)


@pytest.fixture
def split_csv(tmp_path: Path) -> Path:
    path = tmp_path / "1cls.csv"
    path.write_text(CSV_HEADER + CSV_ROWS, encoding="utf-8")
    return path


class TestClassConstants:
    def test_rigid_and_deformable_are_disjoint(self):
        assert not set(RIGID_CLASSES) & set(DEFORMABLE_CLASSES)

    def test_twelve_classes_total(self):
        assert len(ALL_CLASSES) == 12
        assert len(set(ALL_CLASSES)) == 12


class TestLoadSplit:
    def test_reads_every_row(self, split_csv, tmp_path):
        assert len(load_split(split_csv, tmp_path)) == 4

    def test_resolves_image_paths_against_the_dataset_root(self, split_csv, tmp_path):
        sample = load_split(split_csv, tmp_path)[0]
        assert sample.image == tmp_path / "candle/Data/Images/Normal/0836.JPG"

    def test_normal_images_have_no_mask(self, split_csv, tmp_path):
        normals = [s for s in load_split(split_csv, tmp_path) if not s.is_anomalous]
        assert normals and all(s.mask is None for s in normals)

    def test_anomalous_images_carry_a_mask(self, split_csv, tmp_path):
        anomalies = [s for s in load_split(split_csv, tmp_path) if s.is_anomalous]
        assert anomalies and all(s.mask is not None for s in anomalies)

    def test_rigid_only_drops_the_deformable_classes(self, split_csv, tmp_path):
        samples = load_split(split_csv, tmp_path, rigid_only=True)

        assert {s.object_class for s in samples} == {"candle", "pcb1"}
        assert all(s.is_rigid for s in samples)


class TestVerify:
    """Regression guard.

    The archive has no wrapping VisA/ directory, so pointing the root one level too deep
    still parses 10,821 rows and reports healthy counts while every path dangles. Row
    counts are not evidence that the dataset resolved.
    """

    def test_dangling_paths_raise(self, split_csv, tmp_path):
        samples = load_split(split_csv, tmp_path / "wrong-root")

        assert samples, "the bug is that parsing still succeeds"
        with pytest.raises(FileNotFoundError, match="missing on disk"):
            verify(samples)

    def test_empty_split_raises(self):
        with pytest.raises(FileNotFoundError, match="no samples"):
            verify(())

    def test_resolved_paths_pass(self, split_csv, tmp_path):
        samples = load_split(split_csv, tmp_path)
        for sample in samples:
            sample.image.parent.mkdir(parents=True, exist_ok=True)
            sample.image.touch()
            if sample.mask is not None:
                sample.mask.parent.mkdir(parents=True, exist_ok=True)
                sample.mask.touch()

        verify(samples)  # must not raise


class TestSplitCsvPath:
    def test_points_inside_the_archive(self, tmp_path):
        assert split_csv_path(tmp_path) == tmp_path / "split_csv" / "1cls.csv"

    def test_other_setups_are_addressable(self, tmp_path):
        assert split_csv_path(tmp_path, setup="2cls_fewshot").name == "2cls_fewshot.csv"


class TestSample:
    def test_is_rigid_follows_the_documented_scope(self):
        rigid = Sample("pcb1", "test", "anomaly", Path("a.jpg"), Path("m.png"))
        deformable = Sample("cashew", "test", "anomaly", Path("a.jpg"), Path("m.png"))

        assert rigid.is_rigid
        assert not deformable.is_rigid
