# SUMO + PPO: 한국 규칙으로 실제 교통 속 달리기 (`sumo_kr/`)

2D 연습장(`autodrive/`)에서 한 단계 올라가, 무료 오픈소스 교통 시뮬레이터 **[SUMO](https://eclipse.dev/sumo/)** (EPL-2.0)에서
강남역·대치동 **실제 도로망(OpenStreetMap)** 위를 배경 차량들과 함께 달려요. 모두 CPU, Kaggle·Hugging Face 안 씀.

## 설치·실행

```bash
pip install eclipse-sumo libsumo traci sumolib pyproj gymnasium stable-baselines3
python -m sumo_kr.build_net                 # 캐시된 OSM → SUMO 도로망 + 배경 교통 (sumo_kr/build/, git 무시)
python -m sumo_kr.train --eval-only         # 기준 운전자 점수표
python -m sumo_kr.train --budget-minutes 8  # BC(DAgger) → PPO → 평가 → 승격 (다시 돌리면 최고 모델에서 이어 학습)
```

## 어떻게 되어 있나요

- **도로망**: `korea_roads/data/*.json`(OSM 캐시)을 OSM XML로 되돌려 `netconvert`로 변환. 우측 통행(SUMO 기본), OSM 신호 노드로 신호 생성(초록 25초·노랑 3초).
- **배경 교통**: `randomTrips.py`로 2.5초마다 한 대, SUMO 기본 운전자(Krauss)가 몰아요.
- **내 차(ego)**: 경로는 웨이포인트(강남역: 신논현→강남역→역삼역 / 역방향: 역삼→강남역 **우회전**→신논현 / 대치동: 대치초 주변 어린이보호구역)를 잇는 최단 경로.
  `speedMode=0`이라 SUMO가 신호·앞차·제한속도를 대신 지켜 주지 않고, **에이전트가 가감속으로 직접** 지켜야 해요. 경로에 꼭 필요한 차로 변경은 SUMO 기본 전략이 하고, 에이전트는 추가로 왼/오른 차로 변경을 요청할 수 있어요.
- **한국 규칙** (`rules/kr.yaml`): 엣지마다 제한속도를 다시 칠해요 — OSM `maxspeed` 우선, 없으면 안전속도 5030(간선 50 / 이면도로 30 / 도시고속화 60), 초등학교·유치원·어린이집 300 m 안은 어린이보호구역 30 km/h.
  빨간불에 정지선 통과 = 신호 위반, 빨간불 **우회전은 정지선 앞에서 멈춘 뒤에만**(우회전 일시정지, 안 멈추면 위반).
- **관측(14)**: 속도, 제한속도, 앞차 거리·상대 속도, 다음 신호 거리·빨강·노랑, 우회전 여부, 어린이보호구역, 차로 위치, 왼/오른 차로 있음, 경로상 차로 맞추기 필요, 진행률.
- **행동(2)**: 가속(-4.5~+2.6 m/s²), 차로 변경 요청(왼/유지/오른).
- **보상**: 나아간 거리 − 과속(어린이보호구역 2배) − 신호 위반 20 − 우회전 일시정지 위반 10 − 충돌 30 − 급정거, 도착 +10.
- **기준 운전자**: 앞 구간 제한속도 미리 보기 + 앞차 간격 + 빨간불 정지 + 우회전 일시정지 규칙. BC/DAgger의 선생님이자 비교 기준.
- **승격**: 시나리오 3개 × 시드 2개 점수표에서 총점이 오르고, 충돌·신호/우회전 위반·어린이보호구역 과속·완주율이 어디서도 나빠지지 않을 때만 `weights/sumo_best.zip` 교체.

## 결과

SUMO_RESULTS
