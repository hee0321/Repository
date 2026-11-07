"""학습 데이터 라운드 번호·보존 시험 (임시 폴더에서만)."""
import importlib
import json

import numpy as np
import pytest


def _load(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_DATA", str(tmp_path))
    import train_vision
    tv = importlib.reload(train_vision)
    monkeypatch.setattr(tv, "REPO_MANIFEST", str(tmp_path / "repo_manifest.jsonl"))
    monkeypatch.setattr(tv, "DAILY_LOG", str(tmp_path / "daily.jsonl"))
    return tv


def _episode(tv, rnd, i=0, n=5):
    np.savez(tv.os.path.join(tv.DATA, f"r{rnd:02d}_{i:04d}.npz"), img=np.zeros((n, 4, 72, 128), np.uint8),
             nav=np.zeros((n, 30), np.float32), act=np.zeros((n, 3), np.float32), w=np.ones(n, np.float32))


def test_round_ids_never_reused_and_no_overwrite(tmp_path, monkeypatch):
    tv = _load(tmp_path, monkeypatch)
    _episode(tv, 3)
    tv.pack_round(3)
    row = tv.record_round(3)
    assert row["frames"] == 5
    # 디스크에서 사라져도 manifest 가 번호를 기억 → 다음 번호는 4
    for f in tmp_path.glob("round_03_*"):
        f.unlink()
    assert tv.next_round_id() == 4
    assert tv.check_data()[1] == [3]
    # 같은 번호 파일이 있으면 덮어쓰지 않음
    _episode(tv, 4)
    tv.pack_round(4)
    _episode(tv, 4, i=1)
    with pytest.raises(FileExistsError):
        tv.pack_round(4)
    rows = [json.loads(x) for x in open(tmp_path / "MANIFEST.jsonl")]
    assert [r["round"] for r in rows] == [3]
