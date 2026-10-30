"""현재 최고 모델(weights/best.zip)의 '모델+안전 실드' 점수표를 results/best_scorecard_shielded.json 에 저장.
최고 모델이 바뀌었을 때만 다시 잰다 (results/best_scorecard.json 의 from_cycle 비교)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodrive.bench import evaluate  # noqa: E402

best = json.load(open("results/best_scorecard.json", encoding="utf-8"))
out = "results/best_scorecard_shielded.json"
if os.path.exists(out) and json.load(open(out, encoding="utf-8")).get("from_cycle") == best.get("from_cycle") and "--force" not in sys.argv:
    print("실드 점수표 최신 (", best.get("from_cycle"), ")")
    raise SystemExit
c = evaluate("weights/best.zip", shield=True)
c["from_cycle"] = best.get("from_cycle"); c["bench"] = best.get("bench")
json.dump(c, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(f"실드 점수표: 총점 {c['total']} (도시 {c['urban']}, 시골 {c['rural']}), 실드가 바꾼 비율 가속 {c['long_override_share']}, 조향 {c['lat_override_share']}")
