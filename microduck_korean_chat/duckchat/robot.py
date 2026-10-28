"""robotd 에 보낼 JSON-RPC 를 출력만 합니다 (dry-run). 실제 로봇에 연결하지 않음.

실제 로봇에서는 /run/robotd.sock (Unix socket, 줄 단위 JSON-RPC 2.0) 으로 보내면 됩니다 —
microduck_rl/scripts/pickup_e2e_robotd.py 의 rpc()/mover() 와 같은 방식.
"""
from __future__ import annotations

from .intent import Command

SOCKET = "/run/robotd.sock"


def dispatch(cmd: Command | None, dry_run: bool = True) -> list[str]:
    if cmd is None:
        return []
    if not dry_run:
        raise NotImplementedError("이 프로토타입은 dry-run 전용입니다")
    lines = [f"[dry-run → {SOCKET}] {line}" for line in cmd.rpc_lines()]
    if cmd.note:
        lines.append(f"[dry-run] 참고: {cmd.note}")
    return lines
