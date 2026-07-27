# HITL ActionAgent — AMHS LangGraph 챗봇

기존 사내 LangGraph 멀티에이전트 구조(**Router → Supervisor → members → FinalAnswerAgent**)에
**Human-In-The-Loop(HITL)가 적용된 `ActionAgent`** 를 끼워 넣은 구현체입니다.

- 사내 DB/인증은 아직 붙이지 않고 **하드코딩 목업**으로 "HITL이 실제로 도는지"를 검증합니다.
- 모든 설정값은 `.env` 로 분리했습니다 → 사내 값만 채우면 그대로 반입 가능합니다.
- 목업 데이터는 `psns0122/log` 레포의 실제 형식(carrier_id 8자 영숫자, PORT/STOCKER 판정,
  `(reason, description)` 에러 콤보)에 맞춰 만들었습니다.

---

## 빠른 시작

```bash
pip install -r requirements.txt
cp .env.example .env          # FAKE_LLM=1 이면 LLM 없이도 전부 동작합니다

# 터미널 1 — API 서버
uvicorn app.api.main:app --reload --port 8000
# 터미널 2 — 채팅 UI
streamlit run streamlit_app.py
```

검증만 빠르게 하려면:

```bash
python3 tests/test_hitl_scenarios.py   # 그래프 레벨 HITL 시나리오 12종
python3 tests/test_api_sse.py          # SSE API E2E (신규턴/재개/중단/로그)
python3 tests/test_streamlit_ui.py     # Streamlit UI E2E (위젯 조작 → 실제 서버 왕복)
```

> 셋 다 `FAKE_LLM=1` 로 돌아가므로 사내 LLM endpoint 없이 검증됩니다.
> `FAKE_LLM=0`(실제 LLM) 경로는 사내에서 한 번 확인해 주세요.

모델은 **프론트에서 고릅니다** (사이드바 드롭다운). 고른 모델명이 요청에 실려 오고,
서버는 그 모델명을 키로 그래프를 캐싱합니다.

데모 노트북: `notebooks/hitl_demo.ipynb` (셀을 위에서부터 실행)

---

## 아키텍처

### 전체 그래프 (기존 구조 유지 + ActionAgent 삽입)

```
                 ┌─────────┐
   user query ──▶│  START  │
                 └────┬────┘
                      ▼
                 ┌─────────┐   일반질의        ┌──────────────┐  FINISH  ┌───────────────────┐
                 │ Router  │─────────────────▶│ GeneralAgent │────────▶│ FinalGeneralAgent │──▶ END
                 └────┬────┘                   └──────────────┘          └───────────────────┘
                      │ 그 외(supervisor)
                      ▼
                 ┌────────────┐◀───────────── 모든 member 는 실행 후 Supervisor 로 복귀
                 │ Supervisor │──┬─ LocationAgent ─┐
                 │  (배분)     │  ├─ StatusAgent ───┤
                 └─────┬──────┘  ├─ ExtractAgent ──┤──▶ (each) ──▶ Supervisor
                       │         ├─ LogAgent ──────┤
                       │         └─ ActionAgent ★ ─┘   ← HITL 서브그래프
                       │ FINISH / FinalAnswerAgent
                       ▼
                 ┌──────────────────┐
                 │ FinalAnswerAgent │──▶ END   (사용자는 이 노드 토큰을 스트리밍으로 봄)
                 └──────────────────┘
```

### ActionAgent 내부 (⏸ = 질문 남기고 턴 종료)

```
 Supervisor ──▶ [ActionAgent 서브그래프]
        ▼
   ┌────────────┐  entry: 답변 소비 / 복귀 / 신규 판정 (턴 기반의 관제탑)
   │action_entry│
   └─────┬──────┘
         ▼ (신규)
   ┌────────────┐  의도/파라미터/참조 추출 (재진입이면 건너뜀)
   │infer_intent│
   └─────┬──────┘
         ▼
   ┌────────────┐◀───────────────────────────────────┐
   │param_check │ 필수 파라미터 충족? + 헬퍼 결과 회수    │
   └─────┬──────┘                                      │
    ┌────┼──────────┬──────────────┐                   │
    ▼    ▼          ▼              ▼                   │
 ask_    validate  needs_exit   abandon                │
 param⏸    │        (Supervisor    (취소/한도)          │
    │      │         에 상담)                           │
 (턴종료)  │ 실패 → bad 필드만 비우고 ────────────────────┘
    ▼      │          재수집 (MAX_VALIDATE)
 merge_param  ← 다음 턴에 entry 가 답변을 들고 보내줌
    │ 분기: 취소 / 값 / 상담 / 맥락이탈 / 해석불능
    └──▶ param_check
           validate 통과 ▼
                 ┌────────────┐
                 │ask_confirm⏸│  "진짜 실행할까요?" (턴 종료)
                 └────┬───────┘
                      ▼ 다음 턴에 entry → confirm_verdict
              거절 ◀──┴──▶ 승인
                │           │
            abandon      execute  ← 실제 side-effect 는 여기서만
                └────┬──────┘
                     ▼ finalize → END → Supervisor → FinalAnswerAgent
```

---

## HITL 메커니즘 — 턴 기반 (interrupt 를 쓰지 않는 이유)

**모든 사용자 입력은 예외 없이 Router → Supervisor 를 경유한다** — 이 구조
불변식이 이 그래프의 제1 요구사항입니다. LangGraph 의 `interrupt()` +
`Command(resume=)` 는 재개 시 **멈췄던 노드로 직행**하기 때문에 HITL 답변이
Router/Supervisor 를 우회하게 되어, 이 불변식과 양립할 수 없습니다.

그래서 HITL 을 턴 기반 상태 기계로 구현합니다:

1. **질문 = 턴의 정상 종료.** 사용자에게 물을 게 생기면 질문 payload 를
   `action.awaiting` 에 싣고 서브그래프를 끝낸다(`ask_param`/`ask_confirm`).
   Supervisor 가 awaiting 을 보고 턴을 닫는다 (FinalAnswer 없이 END).
2. **답변 = 새 턴.** 사용자의 답은 신규 질문과 똑같이
   `{"messages":[HumanMessage(...)]}` 로 들어온다. Router 가 진행 중 액션을
   보고 Supervisor 로 고정하고(결정적 — LLM 오분류 여지 없음), Supervisor 가
   ActionAgent 로 보내면, entry 가 awaiting + 새 HumanMessage 를 보고 그
   발화를 답변으로 소비한다.
3. **상태의 근거는 오직 `action` 스크래치.** 그래프가 물리적으로 멈춰 있을
   필요가 없으므로, 고아 인터럽트·모델 전환 시 재개 유실·동시 resume 경쟁
   같은 인터럽트 생명주기 문제가 계열째 사라진다.

### 지켜야 하는 규칙

1. **실제 실행(side-effect)은 명시적 승인 뒤 `execute` 노드에만** 둡니다.
   승인 판정(`confirm_verdict`)에서 execute 로 가는 길은 approve 하나뿐입니다.
2. **HITL 대기 감지는 이벤트가 아니라 상태 검사로** 합니다.
   `astream_events` 를 다 돌린 뒤 `aget_state()` 의 `action.awaiting` 을 봅니다.
3. finalize/abandon/restart 에서 **스크래치를 반드시 `{}` 로 리셋**합니다 —
   안 하면 다음 요청이 이전 명령으로 오염됩니다.

---

## 서버 API

### `POST /llm/api/chat/stream`

요청 본문 (`ChatRequest`):

```json
{
  "query": "6PDMQ283 반송해줘",
  "thread_id": "chat-001",
  "model_name": "GaiA-LLM-Latest",
  "recursion_limit": 20
}
```

| 필드 | 기본값 | 설명 |
|---|---|---|
| `query` | (필수) | 사용자 입력. HITL 대기 중이면 그 질문에 대한 답변으로 해석됩니다. |
| `thread_id` | (필수) | 채팅 세션 식별자. 세션당 하나를 유지해야 맥락이 이어집니다. |
| `model_name` | `null` | 프론트에서 고른 모델. `null` 이면 `.env` 기본 모델. |
| `recursion_limit` | `20` | LangGraph recursion limit (`ge=1`, `le=200`). |

**같은 엔드포인트가 신규 질문과 HITL 답변을 모두 처리합니다.** 어느 쪽이든
입력 경로는 동일합니다 — 항상 `{"messages": [HumanMessage(query)]}` 새 턴으로
들어가 Router → Supervisor 를 경유하고, 진행 중 액션이 있으면 Supervisor 가
ActionAgent 로 보냅니다. (HITL 답변이라고 그래프 중간으로 직행하는 일은 없습니다)

#### 스트림 포맷

최종 답변 토큰은 **가공 없는 raw text** 로 그대로 흘립니다(사내 현행 방식).
제어 정보만 텍스트와 섞이지 않게 **`\x1e`(RS) 로 시작하는 JSON 한 줄**로 보냅니다.

```
✅ 반송요청명령 실행 완료          <- 그냥 텍스트 (화면에 그대로)
\x1e{"type":"needs_input",...}\n   <- 제어 프레임
```

`\x1e` 는 일반 텍스트에 나올 일이 없는 제어문자라 안전하게 갈라낼 수 있습니다
(RFC 7464 JSON Text Sequences 와 같은 방식). 클라이언트 구현은 `streamlit_app.py`
의 `stream_chat()` 를 그대로 가져다 쓰면 됩니다.

제어 프레임 종류:

| `type` | 언제 | 내용 |
|---|---|---|
| `node_enter` | 그래프 노드 진입 | `{agent, node}` — 트레이스 창 |
| `tool_call` | 툴 호출 | `{agent, tool, args, result?}` |
| `agent_status` | 에이전트 상태 한 줄 | `{agent, detail}` |
| `needs_input` | **HITL 대기 (턴 종료)** | `{kind: collect_param\|confirm, prompt, field?, action, params, options?}` |
| `thinking` | 중간 에이전트 토큰 (기본 off) | `{agent, text}` — `SHOW_THINKING_TOKENS=1` 일 때만 |
| `usage` | 턴 종료 | 토큰/시간 집계 |
| `done` | 종료 | `{reason: complete\|interrupted\|stopped\|error}` |
| `error` | 오류 | `{message}` |

`needs_input` + `done{interrupted}` 를 받으면 프론트는 입력창/승인버튼을 띄우고,
사용자의 답을 **같은 `thread_id` 로 다시 `/chat/stream`** 에 보내면 됩니다.

### `POST /llm/api/chat/stop`

```json
{ "thread_id": "chat-001" }
```

- **실행 중**: `app.state.stop_flags[thread_id]` 를 세워 스트림 루프를 다음 이벤트에서
  탈출시킵니다.
- **HITL 대기 중 (진행 중 액션)**: 턴 기반이라 그래프는 이미 끝나 있고 `action`
  스크래치만 남아 있습니다. `aupdate_state(config, {"action": {}})` 로 리셋하면
  끝 — 다음 질문은 오염 없이 새로 라우팅됩니다.

### `GET /llm/api/models`

프론트 드롭다운용 모델 목록. 게이트웨이 `/models` 와 같은 형태입니다.

```json
{
  "object": "list",
  "data": [
    {"id": "GaiA-LLM-Latest",       "object": "model", "owned_by": "in-house", "is_default": true},
    {"id": "gaia-GLM-5.2",          "object": "model", "owned_by": "in-house", "is_default": false},
    {"id": "Qwen3.5-397B-A17B-FP8", "object": "model", "owned_by": "in-house", "is_default": false}
  ]
}
```

목록은 `app/_llm.py` 의 `AVAILABLE_MODELS` 에 하드코딩되어 있습니다.
게이트웨이에서 직접 받아오려면 `list_models()` 본문만 바꾸면 됩니다.

### 모델별 그래프 캐싱

LangGraph 빌드는 무거워서 매 요청마다 만들 수 없고, 하나만 만들어 두면 모델 변경이
반영되지 않습니다. 그래서 `app/api/graph_service.py` 가 **모델명을 키로** 캐싱합니다.

```python
_dynamic_graph_cache: Dict[str, dict] = {}   # model_name -> {graph, checkpointer}
_lock = asyncio.Lock()

async def get_team_graph(model_name=None):
    cache_key = model_name if model_name else "default"
    ...
```

체크포인터도 그래프와 짝으로 같이 캐싱합니다 (HITL 재개가 체크포인터에 붙어 있으므로).

---

## 에이전트 간 연계 (needs-핸드오프)

> ActionAgent 를 Router 에 Supervisor 와 **동급**으로 붙이면 다른 에이전트와 협업이 안 됩니다.
> 그래서 Supervisor 밑 member 로 두고, 필요한 값은 **동료에게 잠깐 양보해서** 받아옵니다.

```
"로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘"

 ActionAgent : eqp_id 를 사용자 발화에서 직접 못 읽음
               → action.needs = {question: 내가 물은 것, answer: 사용자 답 원문,
                                 fill: "eqp_id"} 기록
               → interrupt 가 아니라 '정상 종료'로 Supervisor 에 상담
 Supervisor  : 로스터를 보고 "이 답을 풀 수 있는 워커"를 LLM 으로 선정
               (needs_dispatch) + 그 워커에게 보낼 질의문 작성 → 라우팅
 LogAgent    : needs 계약을 모르는 평범한 노드. 대화에 실린 질의를 평소처럼
               처리하고 자연어로 답한다
 Supervisor  : 워커의 답변 원문을 needs_result 메일박스에 실어 ActionAgent 반환
 ActionAgent : 스크래치 생존 → param_check 부터 재개
               → 워커의 답을 사용자 답변 읽듯 ID 판독기로 읽어 파라미터 흡수
               → 사용자에게 묻지 않고 confirm ⏸
```

핵심: **ActionAgent 는 동료의 이름도, 능력도 모릅니다.** 아는 것은
(내가 물은 질문 / 사용자의 답 원문 / 필요한 값) 세 가지뿐이고, 배분은
Supervisor 가 자기 로스터로 판단합니다. 그래서 **워커를 새로 붙이면 배분
프롬프트에 설명 한 줄 추가하는 것만으로 needs 상담 대상에 자동 편입**됩니다
(ActionAgent·워커 본문 수정 없음). HITL 과 대칭 구조이기도 합니다 —
**사람에게 물으면 `awaiting`(턴 종료), 동료에게 물으면 `needs`(턴 내 상담).**

가드: `MAX_HOPS` 왕복 상한 + 상담 실패도 재질문 1회로 계수(`MAX_COLLECT`).
도와줄 워커가 없거나 답에서 값을 못 읽으면 **HITL 로 강등**해 사용자에게 직접 묻습니다.

---

## 액션 도중 탈출

모든 HITL 대기 지점에서 빠져나올 수 있습니다.

- **대화로**: 답변을 값/승인으로 해석하기 **전에** 취소 의도("취소/그만/됐어")를 먼저 검사 → `abandon`
- **버튼/stop**: `/chat/stop` 이 `action` 스크래치를 리셋 → 대기 해제

`abandon` 은 안내 메시지를 남기고 스크래치를 리셋합니다. 거절은 **재시도 루프를 돌지 않습니다.**

---

## 상태 설계 — 왜 `messages` 밖에 두는가

`messages` 에는 최근 4턴만 남기는 커스텀 리듀서가 걸려 있습니다. HITL 문답을 `messages` 에만
두면 **진행 중이던 액션 컨텍스트가 잘려서 깨집니다.**

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], create_limiter(max_turns=4)]
    route: Literal["general", "supervisor"]
    handoff: bool
    next: str
    step: int
    model_name: str                          # 프론트에서 고른 모델
    action: ActionScratch                    # 교체(last-write-wins) — limiter 영향 없음
    facts: Annotated[dict, merge_dict]       # 에이전트 간 공유 팩트
```

`action` 스크래치가 의도/파라미터/검증결과/재시도 카운터/needs 를 들고 있고,
`finalize`/`abandon` 에서 `{}` 로 **반드시 리셋**해 다음 요청이 오염되지 않게 합니다.

---

## 로그

경로: `logs/{env}/{YYYY-MM}/{YYYY-MM-DD}.jsonl` (KST 기준, 최상위 키는 개행 구분·값은 compact)

HITL 때문에 한 액션이 **여러 번의 `/chat/stream`** 에 걸쳐 진행되므로, 토큰/시간은
`thread_id` 별 원장(`app/api/usage_store.py`, in-memory)에 누적하고
**완료 시점에 1건**만 기록합니다.

```json
{
  "thread_id": "chat-001",
  "query": "6PDMQ283 반송해줘",
  "route": "supervisor",
  "outcome": "complete",
  "step_history": ["Router", "Supervisor", "ActionAgent", "Supervisor", "FinalAnswerAgent"],
  "token_cost": { "user_input_tokens": 4, "per_agent": { "Router": {"input": 14, "output": 7} },
                  "total_tokens": 163 },
  "time_cost":  { "ttft_ms": 251, "elapsed_ms": 258, "human_wait_ms": 145, "compute_ms": 113 },
  "hitl":       { "rounds": 2, "stream_calls": 3 }
}
```

`human_wait_ms` 는 사용자가 HITL 질문에 답하기까지 걸린 시간이라, 이를 뺀 `compute_ms` 가
실제 시스템 지연입니다. 읽을 때는 `read_log_records(path)` 를 쓰세요 (한 레코드가 여러 줄이라
줄 단위 파싱이 안 됩니다).

---

## 에이전트 "생각" 보여주기

현재 설정은 **상태·툴 이벤트만** 트레이스로 내보냅니다 (`node_enter`, `tool_call`, `agent_status`).
Streamlit 이 이를 `st.status` 에 흘려서 "Supervisor 판단 중… / LocationAgent 실행 / param_check_tool 호출…"
처럼 보여줍니다.

중간 에이전트의 **토큰 단위** 스트리밍이 필요하면 `.env` 의 `SHOW_THINKING_TOKENS=1` 로 켜면
`thinking` 이벤트가 추가로 흐릅니다.

---

## 모듈 구조

```
app/
├── config.py            # .env 로드
├── _state.py            # AgentState + 4턴 limiter + ActionScratch + model_name
├── _prompt.py           # 에이전트 프롬프트 모음 ← 사내 문구로 갈아끼우는 지점
├── _tool.py             # 에이전트별 툴 (목업)
├── _llm.py              # LLM 팩토리 + 모델 목록 (list_models)
├── _agent.py            # disable_tool_caching / classify_route_with_llm / create_*_agent
├── _node.py             # 노드들 + Supervisor 배분(ExtractAgent 선행, needs 라우팅)
├── _util.py             # message_content_to_text / normalize_route_label / extract_json_object
├── _mcp.py              # MCP 커넥션 매니저 (lifespan 훅, 스텁)
├── _builder.py          # build_team_graph — 기존 배선 + ActionAgent 삽입
├── actions/             # ★ ActionAgent 도메인
│   ├── mock_db.py       #   목업 DB + 공유 조회 함수(LocationAgent 도 재사용)
│   ├── registry.py      #   ActionSpec + ACTION_REGISTRY ← 액션 추가 지점
│   ├── resolvers.py     #   답변 해석 4분기 / 승인 판정
│   ├── tools.py         #   param_check / validate / confirm / execute
│   └── graph.py         #   HITL 서브그래프
└── api/
    ├── main.py          # FastAPI 엔트리 (lifespan: MCP connect/disconnect)
    ├── routes.py        # /chat/stream, /chat/stop, /models, 일별 jsonl 로그
    ├── usage_store.py   # thread_id 별 토큰/시간 원장
    ├── graph_service.py # 모델명 키 그래프 캐시
    └── schemas.py       # ChatRequest / ChatResponse / StopRequest
```

### 에이전트별 툴

에이전트 코드는 전부 같은 모양이고 **붙는 툴만 다릅니다** (`app/_agent.py`).

| 에이전트 | 툴 |
|---|---|
| GeneralAgent | `general_tool`, `amhs_rag_tool` |
| StatusAgent | `queue_status_tool`, `server_status_tool`, `sysadmin_tool`, `patch_plan_search_tool`, `eqp_search_tool` |
| LocationAgent | `location_search_tool` |
| LogAgent | `log_search_tool` |
| ExtractAgent | `fab_extract_tool`, `params_extract_tool` |
| ActionAgent | (HITL 서브그래프가 직접 호출 — `actions/tools.py`) |

모든 툴은 `disable_tool_caching()` 을 거칩니다. 설비/캐리어 상태는 계속 바뀌므로
같은 질문이라도 매번 실제 DB 를 봐야 하기 때문입니다.

### ExtractAgent 는 왜 항상 먼저 도는가

ExtractAgent 는 답변을 내는 워커가 아니라 **뒤 단계가 쓸 ID 재료를 만드는 선행 단계**입니다.
그래서 Supervisor 가 그 턴에 Extract 가 아직 안 돌았으면 무조건 먼저 태웁니다.

```
[NODE] Router entered
[NODE] Supervisor entered
[NODE] Supervisor: ExtractAgent 선행 실행     ← 항상 여기부터
[NODE] ExtractAgent entered
[NODE] Supervisor entered
[NODE] Supervisor -> LocationAgent
...
```

> **"사용자 질의하면 다시 Supervisor 부터 시작하는 게 이상한가?"** — 이상하지 않습니다.
> 워커는 실행 후 항상 Supervisor 로 복귀하는 구조라, 매 턴 Supervisor 가 다시 판단하는 게
> 원래 정상 동작입니다. Extract 선행은 그 위에 얹은 결정적 규칙 하나일 뿐입니다.
>
> 다만 `member_answered_this_turn()` 판정에서는 ExtractAgent 를 **빼야** 합니다
> (`ANSWERING_MEMBERS`). 안 빼면 Extract 가 돌자마자 "워커가 답했다"고 보고 턴이 끝나버립니다.

### 액션 추가하기

`app/actions/registry.py` 에 `ActionSpec` 한 개를 추가하고 툴 3개(validate/confirm/execute)를
쓰면 끝입니다. 서브그래프 배선은 건드릴 필요 없습니다.

```python
ACTION_REGISTRY["hold_carrier"] = ActionSpec(
    name="hold_carrier",
    label="캐리어 홀드",
    required_params=["carrier_id"],
    param_prompts={"carrier_id": "홀드할 carrier_id 를 알려주세요."},
    validate=hold_validate_tool,
    confirm_text=hold_confirm_tool,
    execute=hold_execute_tool,
)
```

---

## ipynb / 서버 / Streamlit 관계

- **노트북에서 API 서버 띄우기**: 됩니다. uvicorn 을 백그라운드 스레드로 올리는 셀이 부록에 있습니다.
- **노트북에서 Streamlit 띄우기**: 안 됩니다. Streamlit 은 자체 프로세스로 떠야 하므로
  터미널에서 `streamlit run streamlit_app.py` 로 실행하세요.
- 따라서 **`.py` 모듈이 소스 원본**이고, 노트북은 그 모듈을 import 해서 검증/시연하는 하니스입니다.

---

## HITL 구현 방식 두 가지 (브랜치 비교)

| | `hitl_new` (이 브랜치) — 턴 기반 | 구 브랜치 — interrupt 기반 |
|---|---|---|
| HITL 답변 경로 | **항상 Router → Supervisor 경유** (새 턴) | 멈춘 노드로 직행 (`Command(resume=)`) |
| 대기 상태 | `action.awaiting` (그냥 데이터) | 체크포인터의 인터럽트 (그래프가 물리적으로 멈춤) |
| 모델 전환 중 답변 | 안전 (상태는 스레드에 붙음) | 체크포인터 공유 필수, 관리 소홀 시 재개 유실 |
| /chat/stop | `action` 스크래치 리셋 한 줄 | abort 센티널로 재개해 abandon 경로 태우기 |
| LangGraph 정합성 | 정석 패턴(interrupt) 아님 | 정석 패턴 |

`notebooks/hitl_demo.ipynb` 는 interrupt 기반(구 브랜치) 기준으로 작성된
데모입니다 — 이 브랜치에서는 `tests/test_hitl_scenarios.py` 가 살아 있는
사용 예시입니다.

---

## 사내 반입 체크리스트

1. `.env` 에 사내 값 채우기 — **변수 이름을 `pptx-vision-rag` 와 동일하게 맞춰뒀으니
   기존 `.env` 의 게이트웨이 설정을 그대로 복사**하면 됩니다.
   ```bash
   FAKE_LLM=0
   LLM_GATEWAY_BASE_URL=http://hcp.llm.skhynix.com/v1
   LLM_GATEWAY_API_KEY=            # 사내 게이트웨이는 키 불필요 → 비워둠
   LLM_CHAT_MODEL=GaiA-LLM-Latest
   LOG_DIR=./devLogs
   ```
   (구 이름 `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`MODEL_NAME` 도 폴백으로 인식합니다.)
2. `app/_llm.py` 의 `get_llm()` 확인 — 이미 사내 게이트웨이 호출 패턴
   (`ChatOpenAI(base_url=…, api_key=… or "EMPTY", max_tokens, max_retries, timeout)`)에
   맞춰져 있습니다. 사내 공용 래퍼(`_llm.llm_t1`)가 따로 있으면 그것으로 교체하세요.
3. `app/actions/mock_db.py` 를 실제 DB 조회로 교체 (함수 시그니처는 그대로 두면 나머지는 무수정)
4. `app/_node.py` 의 Location/Status/Log/Extract 스텁을 사내 실제 노드로 교체
   — 워커에는 needs 관련 코드가 없으므로 **그냥 갈아끼우면 됩니다.**
   needs 상담 배분은 Supervisor(`_agent.needs_dispatch` + 로스터 프롬프트)가 하므로,
   워커를 추가/교체하면 `_prompt.needs_dispatch_prompt` 의 워커 설명만 맞춰 주세요
5. `requirements.txt` 의 langgraph/langchain-core 버전을 사내 버전에 맞추기 (아래 참고)

### 버전 (사내 확인 완료)

사내 실제 버전과 개발·검증 환경이 **동일**합니다. 별도 조정이 필요 없습니다.

| 패키지 | 사내 | 검증 |
|---|---|---|
| `langgraph` | 1.1.2 | 1.1.2 ✅ |
| `langchain-core` | 1.4.9 | 1.4.9 ✅ |
| `langchain-openai` | 1.1.8 | 1.1.8 ✅ |
| `pydantic` | 2.12.5 | 2.12.5 ✅ |
| `langgraph-prebuilt` | — | **1.0.8 필수** ⚠️ |

> ⚠️ **`langgraph-prebuilt` 주의.** langgraph 1.1.2 는 `>=1.0.8,<1.1.0` 을 허용하지만,
> 1.0.9 이상은 `langgraph.runtime` 에서 `ExecutionInfo`/`ServerInfo` 를 import 하는데
> langgraph 1.1.2 에는 그게 없습니다. 그대로 두면 `create_react_agent` import 가
> `ImportError` 로 죽습니다. 그래서 **1.0.8 로 고정**했습니다.
> 사내에서 이미 1.0.9+ 가 깔려 있다면 `pip install langgraph-prebuilt==1.0.8` 로 내려야 합니다.

`requirements.txt` 는 위 버전들을 **정확히 고정(`==`)** 합니다. (턴 기반 전환으로
interrupt API 의존은 사라졌지만, 체크포인터/서브그래프 동작도 버전을 타므로 고정은 유지합니다.) 나머지(fastapi/uvicorn/streamlit/httpx/python-dotenv)는 사내
`pptx-vision-rag` 와 같은 `>=` 하한 방식으로 두었습니다.

> 이 환경에서 `pip install -r requirements.txt` 는 아무것도 바꾸지 않습니다(전부 already satisfied).

턴 기반 전환으로 인터럽트 감지 코드는 사라졌고, HITL 대기 판정은 순수 상태 조회
(`aget_state().values["action"]["awaiting"]`)라 버전 변화에 둔감합니다.
