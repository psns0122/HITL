"""SSE 직렬화.

사내는 현재 event: 필드 없이 data 만 보내는 형태 → 동일하게 data 라인 하나에
JSON 을 싣고, 이벤트 종류는 payload 의 "type" 필드로 구분한다.

이벤트 타입:
- token       : 최종 답변 토큰 (FinalAnswerAgent/FinalGeneralAgent 만)
- node_enter  : 그래프 노드 진입            {agent, node}
- tool_call   : 툴 호출                     {agent, tool, args, result?}
- agent_status: 에이전트 상태 한 줄          {agent, detail}
- needs_input : HITL — 사용자 입력 필요      {kind, prompt, field?, action, params, options?, resume_token}
- usage       : 턴 종료 시 토큰/시간 집계
- done        : 종료 {reason: complete|interrupted|stopped|error}
- error       : 오류 {message}
"""
import json


def sse_pack(payload: dict, seq: int | None = None) -> str:
    if seq is not None:
        payload = {**payload, "seq": seq}
    return "data: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"
