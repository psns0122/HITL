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
`thread_id` 의 진행 상태를 보고 알아서 이어갑니다.

### 응답 — `text/event-stream`

표준 SSE 프레임입니다. 프레임 = `event:` 한 줄 + `data:` 한 줄(JSON) + **빈 줄**.

- `data` JSON 안에 `type` 이 중복으로 들어 있으므로 **event 줄을 무시하고
  data 만 파싱해도 됩니다.**
- 모든 이벤트의 필드는 **항상 전부 실립니다** — 없는 값은 `null`.
  "있을 수도 없을 수도 있는 필드"는 없습니다.
- ⚠ 브라우저 내장 `EventSource` 는 GET 전용이라 못 씁니다. `fetch` 로 POST 하고
  body 스트림을 빈 줄 기준으로 갈라 파싱하세요. (하단 예제)

### 이벤트 — 전부 5개

| event | data 필드 (항상 전부) | 언제 |
|---|---|---|
| `token` | `text` | 최종 답변 생성 중. text 를 이어붙이면 답변 전문 |
| `trace` | `agent`, `tool`, `args`, `result` | 실행 트레이스. `tool=null` 이면 노드 진입, 아니면 툴 실행 |
| `needs_input` | `kind`, `prompt`, `field` | **HITL — 사용자 입력 필요.** 턴 끝에 옴 |
| `usage` | `total_tokens`, `per_agent`, `ttft_ms`, `elapsed_ms`, `compute_ms`, `human_wait_ms`, `hitl_rounds`, `stream_calls` … | 턴 종료 직전 1회. 표시 안 해도 됨 |
| `done` | `reason`, `message` | **항상 마지막 프레임** |

#### `token`

```
event: token
data: {"text": "✅ 반송요청명령 실행 완료\n- Job: TJ-20260728-001"}
```
`text` 를 순서대로 이어붙이면 답변 전문. (개행은 JSON `\n` 으로 옴)

#### `trace`

노드 진입과 툴 실행을 하나의 이벤트로 합쳤습니다. `tool` 로 구분합니다.

```
event: trace
data: {"type":"trace", "agent":"Supervisor", "tool":null, "args":null, "result":null}
```
→ `tool=null` = **노드 진입**. 트레이스 카드에 `▶ Supervisor` 한 줄.
마지막 노드 진입이 `FinalAnswerAgent`(업무 답변) 또는 `FinalGeneralAgent`(일반 대화)라
어느 쪽이 답했는지 구분됩니다.

```
event: trace
data: {"type":"trace", "agent":"ActionAgent", "tool":"param_check_tool", "args":{"action":"transport"}, "result":null}

event: trace
data: {"type":"trace", "agent":"ActionAgent", "tool":"param_check_tool", "args":null, "result":"missing=['eqp_id']"}
```
→ `tool!=null` = **툴 실행**. 시작 프레임에 `args`(입력), 종료 프레임에 `result`(결과)가
따로 옵니다. 한 줄로 보여주려면: `result` 만 있는 프레임이 오면 같은 `tool` 의
"결과가 아직 안 붙은" 줄을 뒤에서부터 찾아 이어 붙이세요.
(노드가 직접 보내는 trace 는 `args` 와 `result` 가 한 프레임에 같이 올 수도 있음 — 그냥 한 줄)

#### `needs_input`

```
event: needs_input
data: {"type":"needs_input", "kind":"collect_param", "prompt":"목적지 장비 ID를 알려주세요. (예: STK102)", "field":"eqp_id"}

event: needs_input
data: {"type":"needs_input", "kind":"confirm", "prompt":"⚠️ 반송요청명령 실행 확인\n- 캐리어: 6PDMQ283 (현재 위치 STK101)\n- 목적지: STK102\n이 명령을 정말 실행할까요? (승인/거절)", "field":null}
```

| kind | 뜻 | 권장 UI |
|---|---|---|
| `collect_param` | 파라미터 값 질문. `field` = 묻는 파라미터 이름 | `prompt` 를 봇 말풍선으로, 입력창 placeholder 를 `field` 기준으로. 자유 입력 허용 ("9ZXCV456 있는 위치로" 같은 간접 답도 서버가 해석) |
| `confirm` | 실행 최종 승인. `field` = null | `prompt` 를 봇 말풍선으로 + [승인] [거절] 버튼. 버튼도 결국 `query="승인"` 전송일 뿐 |

확정된 파라미터 값들은 `prompt` 문구 안에 이미 들어 있습니다 — 별도 필드 없음.
취소는 언제든 자유 입력("취소", "그만")으로 가능 — 서버가 판정합니다.

#### `usage`

```
event: usage
data: {"type":"usage", "total_tokens":1834, "ttft_ms":812, "elapsed_ms":5210, "hitl_rounds":2, "per_agent":{"Router":{"input":120,"output":8,"calls":1}, ...}, ...}
```
턴 종료 직전 1회. "응답 상세" 카드용 — 안 그려도 동작에는 지장 없습니다.

#### `done`

```
event: done
data: {"type":"done", "reason":"interrupted", "message":null}
```

| reason | 뜻 | 프론트가 할 일 |
|---|---|---|
| `complete` | 정상 완료 | 턴 종료 |
| `interrupted` | **HITL 대기** — 직전에 `needs_input` 이 반드시 옴 | 다음 사용자 입력을 그대로 `query` 로 재전송 |
| `stopped` | `/chat/stop` 으로 중단됨 | 턴 종료 표시 |
| `error` | 오류 — `message` 에 사유 | `message` 표시 |

`message` 는 `reason=error` 일 때만 값이 있고 나머지는 `null` 입니다.
(별도의 error 이벤트는 없습니다 — done 에 합쳐져 있습니다)

### 시나리오별 프레임 순서

**A. 일반 완료 (조회 질문)**
```
trace(Router) → trace(Supervisor) → trace(ExtractAgent) → trace(tool…)
→ trace(LocationAgent) → trace(tool…) → trace(Supervisor)
→ trace(FinalAnswerAgent) → token × N → usage → done(complete)
```

**B. HITL 3턴 (명령 실행)**
```
턴1  query="6PDMQ283 반송해줘"
     trace… → needs_input(collect_param, field=eqp_id) → usage → done(interrupted)
턴2  query="STK102"
     trace… → needs_input(confirm) → usage → done(interrupted)
턴3  query="승인"
     trace… → token × N (실행 결과 답변) → usage → done(complete)
```

**C. 일반 대화**
```
trace(Router) → trace(GeneralAgent) → trace(FinalGeneralAgent)
→ token × N → usage → done(complete)
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

      let data = null;
      for (const line of block.split("\n"))
        if (line.startsWith("data:")) data = JSON.parse(line.slice(5));
      if (!data) continue;

      if (data.type === undefined) onToken(data.text);   // token 은 type 없이 {text}
      else onEvent(data);                                // data.type 으로 분기
    }
  }
}
```

> `token` 의 data 는 `{"text": ...}` 뿐이라 `type` 이 없습니다. event 줄로
> 구분하고 싶으면 `event:` 줄을 함께 파싱하세요 (`event: token`).

파이썬 참고 구현: `streamlit_app.py` 의 `stream_chat()`.

---

## 2. POST /chat/stop — 중단/취소

```json
요청:  { "thread_id": "ui-a1b2c3d4" }
```

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
**대화 도중 모델을 바꿔도 됩니다** — 같은 `thread_id` 면 맥락과 HITL 진행 상태가 이어집니다.

---

## 4. GET /health — 헬스체크

```json
{ "ok": true, "default_model": "GaiA-LLM-Latest", "time": "2026-07-28T14:30:00.000000" }
```

`ok: true` 만 보면 됩니다.

---

## 프론트 구현 체크리스트

- [ ] `thread_id` 를 채팅 세션당 하나 생성해 유지 ("새 대화" = 새 `thread_id`)
- [ ] `fetch` POST 로 SSE 수신 (내장 `EventSource` 금지 — GET 전용)
- [ ] `token` → 답변 버블 스트리밍
- [ ] `trace` → 트레이스 카드 (`tool=null` 은 ▶ 노드, 아니면 툴 한 줄 — result 병합)
- [ ] `needs_input` + `done(interrupted)` → prompt 표시, 다음 입력을 그대로 `query` 로 재전송
- [ ] `kind=confirm` 이면 승인/거절 버튼 (자유 입력 병행)
- [ ] `done(error)` → `message` 표시
- [ ] 중단 버튼 → `/chat/stop`, 응답 `mode` 에 따라 HITL 상태 정리
