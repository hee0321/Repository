#!/usr/bin/env bash
# 매일 강화학습(개선 사이클) 루틴: 사이클 하나 = 학습 ≤ ~5분 + 평가 → 끝날 때마다 점수표·리더보드 커밋·푸시.
# 사용: tools/rl_daily.sh [사이클 수] [추가 improve.py 옵션...]
# 가중치(weights/)는 git 에 올리지 않는다 (.gitignore).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
N=${1:-1}; shift || true
BRANCH=${BRANCH:-autodrive/rl-daily}
for i in $(seq 1 "$N"); do
  echo "=== 사이클 $i/$N 시작 $(date '+%F %T')"
  if [ "${MODE:-boost}" = boost ]; then   # 기본: 병렬 DAgger 부스트 (boost.py), MODE=improve 면 예전 improve.py 사이클
    timeout 600 $PY boost.py "$@" || echo "사이클 실패/시간 초과 (exit $?)"
  else
    timeout 600 $PY improve.py --rural --cycles 1 "$@" || echo "사이클 실패/시간 초과 (exit $?)"
  fi
  $PY tools/shield_card.py || true
  git add results/ >/dev/null 2>&1
  git commit -qm "RL 사이클 결과 $(date '+%F %H:%M') ($i/$N)" && git push -q origin "$BRANCH" || echo "커밋할 것 없음/푸시 실패"
done
