# 챗봇 백엔드 API 인터페이스 명세

프론트엔드 개발자에게 전달하는 문서입니다.
백엔드는 API 4개를 제공하고, 이 중 프론트가 반드시 구현해야 하는 것은
**`/chat/stream` 하나**입니다. 나머지는 부가 기능입니다.

- Base URL: `http://{host}:8000/llm/api`
- 인증: 없음 (사내망)
- 문자셋: UTF-8

| # | 메서드 | 경로 | 방식 | 용도 |
|---|---|---|---|---|
| 1 | POST | `/chat/stream` | **SSE 스트리밍** | 질문 전송 + 응답 수신. HITL 답변도 여기로 |
| 2 | POST | `/chat/stop` | JSON 요청/응답 | 실행 중단 / 진행 중 명령 취소 |
| 3 | GET | `/models` | JSON 응답 | 모델 선택 드롭다운용 목록 |
| 4 | GET | `/health` | JSON 응답 | 서버 상태 확인 |

---

## 1. POST /chat/stream — 대화 (SSE)

### 요청

```json
{
  "query": "6PDMQ283 반송해줘",
  "thread_id": "ui-a1b2c3d4",
  "model_name": "GaiA-LLM-Latest",
  "recursion_limit": 20
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `query` | string | ✅ | 사용자 입력. **HITL 답변("STK102", "승인")도 그대로 이 필드로** |
| `thread_id` | string | 권장 | 채팅 세션 식별자. 세션당 하나를 만들어 계속 재사용해야 대화 맥락이 이어짐. 생략 시 `"local_test"` |
| `model_name` | string \| null | — | 사용자가 고른 모델. 생략 시 서버 기본 모델 |
| `recursion_limit` | int (1~200) | — | 그래프 순환 상한. 기본 20. 범위 밖이면 422 |

**신규 질문과 HITL 답변의 요청 형식이 완전히 동일합니다.**
프론트는 "지금이 HITL 대기인지" 를 기억할 필요가 없습니다 — 서버가
`thread_id` 의 진행 상태를 보고 알아서 이어갑니다. (다만 UX 를 위해
마지막 `needs_input` 을 기억해 입력창 placeholder / 승인 버튼을 바꾸는 것을 권장)

### 응답 — `text/event-stream`

표준 SSE 프레임입니다. 프레임 = `event:` 한 줄 + `data:` 한 줄(JSON) + **빈 줄**.

```
event: token
data: {"text": "✅ 반송요청명령"}

event: needs_input
data: {"type": "needs_input", "kind": "confirm", "prompt": "...", "params": {"carrier_id": "6PDMQ283", "eqp_id": "STK102"}}

event: done
data: {"type": "done", "reason": "interrupted"}
```

- **답변 토큰은 `event: token`**, `data.text` 를 순서대로 이어붙이면 답변 전문이 됩니다.
  (text 가 JSON 으로 감싸진 이유: SSE data 줄은 개행을 못 담기 때문. 개행은 `\n` 으로 옵니다)
- 제어 프레임은 `type` 값이 그대로 event 이름이며, `data` JSON 안에도 `type` 이
  중복으로 들어 있습니다. → **event 줄을 무시하고 data 만 파싱해도 됩니다.**
- ⚠ **브라우저 내장 `EventSource` 는 GET 전용이라 못 씁니다.** `fetch` 로 POST 하고
  body 스트림을 빈 줄 기준으로 갈라 파싱하세요. (문서 하단 예제 코드)

### 이벤트 카탈로그

| event | data 필드 | 언제 | 프론트가 할 일 |
|---|---|---|---|
| `token` | `text` | 최종 답변 생성 중 | 답변 버블에 이어붙이기 |
| `node_enter` | `agent`, `node` | 그래프 노드 진입 | 트레이스 카드에 `▶ {agent}` 한 줄. 마지막에 `FinalAnswerAgent`(업무) 또는 `FinalGeneralAgent`(일반 대화)가 찍혀 어느 쪽이 답했는지 구분됨 |
| `tool_call` | `agent`, `tool`, `args?`, `result?` | 툴 실행 | 트레이스 카드에 `- {tool} 입력 → 결과` 한 줄. **아래 "tool_call 병합 규칙" 필독** |
| `needs_input` | `kind`, `prompt`, `action`, `params`, `field?`, `missing?`, `options?` | **HITL — 사용자 입력 필요** | `prompt` 를 봇 말풍선으로 표시. `kind` 별 UI 는 아래 표 |
| `usage` | `total_tokens`, `per_agent`, `ttft_ms`, `elapsed_ms`, `compute_ms`, `human_wait_ms`, `hitl_rounds`, `stream_calls` … | 턴 종료 직전 1회 | (선택) 응답 상세 카드 |
| `error` | `message` | 서버 오류 | 오류 표시 |
| `done` | `reason`, `step_history?` | **항상 마지막 프레임** | `reason` 에 따라 턴 마무리 (아래) |
| `agent_status` | `agent`, `detail` | 에이전트 상태 문구 | **무시해도 됨** (참고용) |
| `thinking` | `agent`, `text` | 중간 에이전트 토큰 | **무시해도 됨** (서버 `.env SHOW_THINKING_TOKENS=1` 일 때만 옴) |

#### done.reason

| reason | 뜻 | 프론트가 할 일 |
|---|---|---|
| `complete` | 정상 완료 | 턴 종료. 입력창 평소대로 |
| `interrupted` | **HITL 대기** — 직전에 `needs_input` 이 반드시 옴 | 사용자 입력을 받아 다음 `/chat/stream` 의 `query` 로 그대로 전송 |
| `stopped` | `/chat/stop` 으로 중단됨 | 턴 종료 표시 |
| `error` | 오류 종료 — 직전에 `error` 프레임이 옴 | 오류 표시 |

#### needs_input.kind

| kind | 뜻 | 권장 UI |
|---|---|---|
| `collect_param` | 파라미터 값 질문 (`field` 에 무엇을 묻는지) | 입력창 placeholder 를 `field` 값 요청으로. 자유 입력 허용 (예: "9ZXCV456 있는 위치로" 같은 간접 답도 서버가 해석) |
| `confirm` | 실행 최종 승인 (`params` 에 확정된 값) | [승인] [거절] 버튼 + 자유 입력 병행. 버튼도 결국 `query="승인"` 전송일 뿐 |

취소는 언제든 자유 입력("취소", "그만")으로 가능 — 서버 LLM 이 판정합니다.

#### tool_call 병합 규칙 (중요)

툴 하나가 **프레임 두 개**로 옵니다. 시작 시점에 입력이, 종료 시점에 결과가 따로 옵니다.

```
event: tool_call
data: {"type":"tool_call", "agent":"ActionAgent", "tool":"param_check_tool", "args":{...}}     ← 시작(입력)

event: tool_call
data: {"type":"tool_call", "agent":"ActionAgent", "tool":"param_check_tool", "result":"..."}    ← 종료(결과)
```

한 줄로 보여주려면: **`result` 만 있고 `args` 가 없는 프레임**이 오면, 같은
`tool` 이름의 "아직 결과가 안 붙은" 줄을 뒤에서부터 찾아 결과를 이어붙이면 됩니다.
(단, 노드가 직접 보내는 tool_call 은 `args` 와 `result` 가 한 프레임에 같이 올 수도
있음 — 그 경우는 그냥 한 줄로 그리면 됨)

### 시나리오별 프레임 순서

**A. 일반 완료 (조회 질문)**
```
node_enter(Router) → node_enter(Supervisor) → node_enter(ExtractAgent)
→ tool_call… → node_enter(LocationAgent) → tool_call…
→ node_enter(FinalAnswerAgent) → token × N → usage → done(complete)
```

**B. HITL 3턴 (명령 실행)**
```
턴1  query="6PDMQ283 반송해줘"
     트레이스… → needs_input(collect_param, field=eqp_id) → usage → done(interrupted)
턴2  query="STK102"
     트레이스… → needs_input(confirm, params={carrier_id, eqp_id}) → usage → done(interrupted)
턴3  query="승인"
     트레이스… → token × N (실행 결과 답변) → usage → done(complete)
```

**C. 일반 대화**
```
node_enter(Router) → node_enter(GeneralAgent)
→ node_enter(FinalGeneralAgent) → token × N → usage → done(complete)
```

### 프론트 파서 예제 (JS / fetch)

```javascript
async function chatStream(query, threadId, onToken, onEvent) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, thread_id: threadId }),
  });

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;

    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {        // 프레임 = 빈 줄 종료
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      let event = "message", data = null;
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data = JSON.parse(line.slice(5));
      }
      if (!data) continue;

      if (event === "token") onToken(data.text);
      else onEvent(data);                                 // data.type 으로 분기해도 됨
    }
  }
}
```

파이썬 참고 구현: `streamlit_app.py` 의 `stream_chat()`.

---

## 2. POST /chat/stop — 중단/취소

```json
요청:  { "thread_id": "ui-a1b2c3d4" }
```

두 상황을 서버가 알아서 구분합니다.

| 상황 | 서버 동작 | 응답 |
|---|---|---|
| 스트림 실행 중 | 중단 플래그 → 해당 스트림이 `done(stopped)` 로 닫힘 | `{"ok": true, "mode": "stop_flag", "thread_id": "..."}` |
| HITL 대기 중 (진행 중 명령 있음) | 명령 진행 상태 리셋 — 다음 질문은 깨끗하게 시작 | `{"ok": true, "mode": "aborted_action", "thread_id": "..."}` |

프론트는 "중단" 버튼 하나로 둘 다 이 API 를 부르면 됩니다.
`mode: "aborted_action"` 을 받으면 기억해 둔 `needs_input` 상태를 버리세요.

---

## 3. GET /models — 모델 목록

```json
{
  "object": "list",
  "data": [
    { "id": "GaiA-LLM-Latest", "object": "model", "owned_by": "in-house", "is_default": true },
    { "id": "gaia-GLM-5.2",    "object": "model", "owned_by": "in-house", "is_default": false }
  ]
}
```

드롭다운은 `data[].id` 로 채우고, 선택값을 `/chat/stream` 의 `model_name` 으로 보냅니다.
`is_default: true` 인 항목을 초기 선택값으로 쓰면 됩니다.
**대화 도중 모델을 바꿔도 됩니다** — 같은 `thread_id` 면 맥락과 HITL 진행 상태가 이어집니다.

---

## 4. GET /health — 헬스체크

```json
{ "ok": true, "default_model": "GaiA-LLM-Latest", "time": "2026-07-28T14:30:00.000000" }
```

`ok: true` 만 보면 됩니다. 초기 로딩 때 한 번 찔러 백엔드 기동 여부를 확인하는 용도.

---

## 프론트 구현 체크리스트

- [ ] `thread_id` 를 채팅 세션당 하나 생성해 유지 ("새 대화" = 새 `thread_id`)
- [ ] `fetch` POST 로 SSE 수신 (내장 `EventSource` 금지 — GET 전용)
- [ ] `token` → 답변 버블 스트리밍
- [ ] `node_enter` / `tool_call` → 트레이스 카드 (tool_call 병합 규칙 적용)
- [ ] `needs_input` + `done(interrupted)` → prompt 표시, 다음 입력을 그대로 `query` 로 재전송
- [ ] `kind=confirm` 이면 승인/거절 버튼 (자유 입력 병행)
- [ ] 중단 버튼 → `/chat/stop`, 응답 `mode` 에 따라 HITL 상태 정리
- [ ] `agent_status` / `thinking` 은 무시 가능
