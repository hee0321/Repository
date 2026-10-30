# satellite/ — 브이월드 위성영상 배경 (국내 실제 도로 위에 겹치기)

국토교통부 **브이월드(VWorld) 영상지도** 타일을 받아 이어 붙여서
① 우리 matplotlib 렌더러(OSM 경로와 같은 좌표)와 ② SUMO GUI 배경(decal)에 정확히 겹친다.

| 항목 | 내용 |
|---|---|
| 출처/라이선스 | 국토교통부 브이월드 WMTS/TMS API, **공공누리 제1유형**(출처표시만 하면 상업적 이용·변형 가능), 무료 ([공공데이터포털](https://www.data.go.kr/data/3046388/openapi.do)) |
| 해상도 | 줌 6~19, 강남 기준 z17 ≈ 0.95 m/px, z18 ≈ 0.47 m/px, z19 ≈ 0.24 m/px |
| 범위 | 대한민국 전역 |
| 키 | 무료 인증키 필요 → 환경변수 `VWORLD_API_KEY` (코드·저장소에 넣지 않음). 웹용 키라면 `VWORLD_DOMAIN`(등록 도메인)도 |
| 출처 표시 | 그림 오른쪽 아래와 SUMO 설정 XML 주석에 자동으로 넣음 |

## 사용

```bash
export VWORLD_API_KEY=발급받은키      # https://www.vworld.kr → 오픈API → 인증키 발급
python satellite_demo.py --out media/satellite_gangnam.png            # 강남역, z17
python satellite_demo.py --zoom 18 --net sumo_kr/build/gangnam.net.xml  # + SUMO decal
sumo-gui -n sumo_kr/build/gangnam.net.xml -g satellite/cache/sumo/gangnam_station.gui.xml
```

키가 없으면 네트워크 요청 없이 타일 격자 자리표시로 범위·정렬만 확인한다.

## 구성

- `tiles.py` — XYZ(z/y/x) 타일 받기. `satellite/cache/`(git 무시)에 캐시, 요청 간격 0.2 s, 한 번에 최대 400장, 한국 범위 밖·키 오류(XML 응답) 검사
- `mosaic.py` — 위경도 범위 → 모자이크 PNG + `.pgw` 월드파일(EPSG:3857) + `.json`(범위·출처)
- `overlay.py` — `draw_satellite(ax, mosaic, lat0, lon0)`: `osm.to_local` 좌표로 정렬(메르카토르 행을 위도 등간격으로 다시 뽑음)
- `sumo_decal.py` — `.net.xml`의 `<location>`(netOffset, UTM 투영)을 읽어 이미지를 망 좌표 격자로 다시 뽑고 `<decal>` gui-settings 생성 (강남 정합 잔차 약 4 cm)

## 규칙

- 타일·모자이크·decal 이미지는 **커밋하지 않는다** (`satellite/cache/`). 필요하면 각자 키로 다시 받는다.
- 브이월드 서버 배려: 캐시 재사용, 초당 5장 이하, 큰 범위는 줌을 낮춘다.
- 네이버·카카오·구글·Esri·Mapbox 위성지도는 유료이거나 약관상 이런 활용이 안 돼서 쓰지 않는다.
