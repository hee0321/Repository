# 연구자동화 에이전트 작업 기록

그동안 에이전트와 진행한 주요 작업 및 설정 내역을 요약하여 기록합니다.

## 1. Luna Music Agent 및 Kodari 스킬 설정
- **작업 내용**: 음악 생성 자동화를 위한 "Luna" 뮤직 에이전트 개발 (lyria_generator.py 등)
- **프롬프트 구성**: Luna_Current_Prompt.txt 를 최적화하여 24/7 YouTube Live Stream 환경에 맞는 음악 생성 기능 고도화
- **스킬 디렉토리**: 커스텀 에이전트 기능 추가를 위한 `.agent/skills` 및 `.agent/skills/kodari` 디렉토리 구조 설정

## 2. NotebookLM 및 MCP(Model Context Protocol) 연동
- **작업 내용**: Google NotebookLM과 MCP 서버를 연결하여 AI 에이전트가 문서를 상호작용할 수 있도록 구성
- **Kodari 페르소나**: MCP 서버 설치 및 요구사항 구성을 가이드 학습 프로세스를 통해 진행

## 3. PC 환경 최적화 및 트러블슈팅
- **화면 자동 꺼짐 방지**: 자동화 작업 중 끊김이 없도록 Windows 전원 및 절전 모드 설정 조정 (화면 꺼짐 및 화면 보호기 비활성화)
- **BSOD 오류 해결**: 0x00000116(그래픽 드라이버) 관련 블루스크린(BSOD) 진단 및 배터리 전력 효율 최적화 방안 적용
