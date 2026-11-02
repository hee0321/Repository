#!/usr/bin/env bash
# 비전 정책 매일 DAgger 라운드 1회 (~25-30분 제한): 모으기(DCAS 25% 섞기) → 1에포크 학습 → v2.2 + DCAS 평가
# → 관문(안전·자율 완주·점수) → 승격 시 weights/vision_best.pt 교체 → 점수표 커밋·푸시(autodrive/vision-sat).
#
# 사용: tools/vision_daily.sh [--budget-min 28] [--episodes 280] [--max-samples 1500000] [--workers 6] [--no-push]
#   짧은 시험: tools/vision_daily.sh --budget-min 12 --episodes 8 --max-samples 4000 --no-push
set -euo pipefail
BUDGET=28; EP=280; MAXS=1500000; W=6; PUSH=1; BRANCH=autodrive/vision-sat
while [ $# -gt 0 ]; do case "$1" in
  --budget-min) BUDGET=$2; shift 2;; --episodes) EP=$2; shift 2;; --max-samples) MAXS=$2; shift 2;;
  --workers) W=$2; shift 2;; --no-push) PUSH=0; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # autonomous_driving/
cd "$HERE"
PY=${PY:-/workspace/autodrive-pov/venv/bin/python}
TMP=${VISION_TMP:-/workspace/vis-tmp}/daily; mkdir -p "$TMP"
exec 9>"$TMP/lock"; flock -n 9 || { echo "다른 vision_daily 실행 중"; exit 3; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "브랜치가 $BRANCH 가 아님"; exit 4; }
T0=$(date +%s); DEADLINE=$((T0 + BUDGET * 60))
left() { echo $((DEADLINE - $(date +%s))); }
log() { echo "[$(date '+%H:%M:%S KST')] $*"; }

[ -f weights/vision_best.pt ] || cp weights/vision_policy.pt weights/vision_best.pt
# 최고 모델 점수가 없으면 먼저 만든다(첫 실행 한 번)
if [ ! -f results/vision_best_v22.json ] || [ ! -f results/vision_best_dcas.json ]; then
  log "최고 모델 기준 점수 만들기"
  rm -f "$TMP/b22.json" "$TMP/bdc.json"
  $PY eval_vision.py --only vision --policy weights/vision_best.pt --workers "$W" --out "$TMP/b22.json" > "$TMP/b22.log"
  $PY eval_vision.py --suite dcas --only vision --policy weights/vision_best.pt --workers "$W" --out "$TMP/bdc.json" > "$TMP/bdc.log"
  cp "$TMP/b22.json" results/vision_best_v22.json; cp "$TMP/bdc.json" results/vision_best_dcas.json
fi

# 1) 모으기 + 학습 (남은 시간의 75% 까지)
TT=$(( $(left) * 75 / 100 )); [ "$TT" -gt 60 ] || { log "시간 부족"; exit 5; }
log "1) DAgger 모으기 $EP 에피소드 + 학습 (최대 ${TT}s)"
rm -f weights/vision_cand.pt
timeout "$TT" $PY train_vision.py --daily --episodes "$EP" --workers "$W" --epochs 1 --max-samples "$MAXS" \
  --dcas-frac 0.25 | tee "$TMP/train.log"
[ -f weights/vision_cand.pt ] || { log "후보 없음(시간 초과?)"; exit 6; }
RND=$(tail -1 "$TMP/train.log" | $PY -c "import json,sys;print(json.loads(sys.stdin.read())['round'])")

# 2) 평가 (v2.2 18 + DCAS 20)
TE=$(left); [ "$TE" -gt 30 ] || { log "평가 시간 부족"; exit 7; }
log "2) 후보 평가 (남은 ${TE}s)"
rm -f "$TMP/c22.json" "$TMP/cdc.json"
timeout "$TE" $PY eval_vision.py --only vision --policy weights/vision_cand.pt --workers "$W" --out "$TMP/c22.json" > "$TMP/c22.log"
timeout "$(left)" $PY eval_vision.py --suite dcas --only vision --policy weights/vision_cand.pt --workers "$W" --out "$TMP/cdc.json" > "$TMP/cdc.log"

# 3) 관문
log "3) 관문"
set +e
$PY tools/vision_gate.py --best-v22 results/vision_best_v22.json --best-dcas results/vision_best_dcas.json \
  --cand-v22 "$TMP/c22.json" --cand-dcas "$TMP/cdc.json" --round "$RND" \
  --log results/vision_daily_log.jsonl --board results/vision_leaderboard.md --note "ep=$EP max_samples=$MAXS" | tee "$TMP/gate.log"
G=${PIPESTATUS[0]}; set -e
if [ "$G" = 0 ]; then
  log "승격: r$RND → weights/vision_best.pt"
  cp weights/vision_cand.pt weights/vision_best.pt
  cp weights/vision_cand.pt "weights/vision_policy_r$RND.pt"
  cp "$TMP/c22.json" results/vision_best_v22.json; cp "$TMP/cdc.json" results/vision_best_dcas.json
elif [ "$G" = 1 ]; then log "유지: 후보 버림"; else log "관문 오류 $G"; exit 8; fi

# 4) 점수표 커밋·푸시 (가중치는 .gitignore 로 제외)
git add results/vision_leaderboard.md results/vision_daily_log.jsonl results/vision_best_v22.json results/vision_best_dcas.json
git commit -qm "vision daily r$RND: $( [ "$G" = 0 ] && echo promoted || echo kept ) ($(date '+%Y-%m-%d %H:%M KST'))" || true
if [ "$PUSH" = 1 ]; then git push -q origin "$BRANCH"; log "푸시 완료"; else log "--no-push: 푸시 생략"; fi
log "끝 ($(( $(date +%s) - T0 ))s)"
