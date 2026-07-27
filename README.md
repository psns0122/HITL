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
uvicorn app.main:app --reload --port 8000
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

### ActionAgent 내부 (⏸ = interrupt 지점)

```
 Supervisor ──▶ [ActionAgent 서브그래프]
        ▼
   ┌────────────┐  의도/파라미터/참조 추출 (재진입이면 건너뜀)
   │infer_intent│
   └─────┬──────┘
         ▼
   ┌────────────┐◀───────────────────────────────────┐
   │param_check │ 필수 파라미터 충족? + 헬퍼 결과 회수    │
   └─────┬──────┘                                      │
    ┌────┼──────────┬──────────────┐                   │
    ▼    ▼          ▼              ▼                   │
 collect  validate  needs_exit   abandon                │
 _param⏸   │        (동료에게      (취소/한도)            │
    │      │         양보)                              │
    ▼      │ 실패 → bad 필드만 비우고 ────────────────────┘
 merge_param          재수집 (MAX_VALIDATE)
    │ 4분기: 취소 / 리터럴 / 참조(needs) / 해석불능
    └──▶ param_check
           validate 통과 ▼
                    ┌──────────┐
                    │ confirm ⏸│  "진짜 실행할까요?"
                    └────┬─────┘
              거절 ◀─────┴─────▶ 승인
                │                │
            abandon           execute  ← 실제 side-effect 는 여기서만
                └────────┬────────┘
                         ▼ finalize → END → Supervisor → FinalAnswerAgent
```

---

## HITL 메커니즘 — 5개 후보 중 무엇을 썼나

| 후보 | 실존 여부 | 판정 |
|---|---|---|
| `interrupt_before` | ⭕ 정적 interrupt | 노드 **앞**에서만 멈춤. 페이로드를 못 실어 "무엇이 부족한지" 전달 불가 → 부적합 |
| `interrupt_after` | ⭕ 정적 interrupt | 노드 **뒤**에서만. 마찬가지로 부적합 |
| **`interrupt()` + `Command(resume=)`** | ⭕ 동적 | **채택.** 노드/툴 내부에서 멈추고 구조화 페이로드 전달·재개 |
| `Command.PAUSE` | ❌ 존재하지 않음 | `Command` 의 멤버는 `resume`/`update`/`goto` 뿐입니다. 폐기 |
| `__interrupt__` 상태 | △ 절반만 맞음 | 트리거가 아니라 인터럽트가 **표면화되는 키**. 감지용으로만 사용 |

> `raise NodeInterrupt(...)` 는 deprecated 라 쓰지 않았습니다.

### 반드시 지켜야 하는 재개 규칙 (버전 민감)

1. **interrupt 된 노드는 resume 시 맨 위부터 재실행됩니다.**
   → interrupt 위쪽에 side-effect(실제 액션·외부 쓰기·LLM 호출)를 두면 재개마다 반복 실행/이중 과금됩니다.
   → 이 구현은 실제 실행을 승인 이후 `execute` 노드에만 두어 구조적으로 회피합니다.
2. 한 노드에 interrupt 가 여러 개면 **실행 순서(positional)** 로 resume 값이 매칭됩니다.
   → `collect_param` / `confirm` **노드당 정확히 1개**만 두었습니다.
3. **인터럽트 감지는 이벤트가 아니라 상태 검사로** 합니다.
   `astream_events` 를 다 돌린 뒤 `aget_state()` 의 `interrupts` 를 보는 것이 정석입니다.

---

## 서버 API

### `POST /llm/api/chat/stream`

```json
{ "query": "6PDMQ283 반송해줘", "thread_id": "chat-001" }
```

**같은 엔드포인트가 신규 질문과 HITL 답변을 모두 처리합니다.** 서버가 체크포인터 상태를 보고
판정합니다.

- 인터럽트 없음 → `{"messages": [HumanMessage(query)]}` 로 신규 턴
- 인터럽트 있음 → `Command(resume=query)` 로 재개

SSE 이벤트 (`data:` 한 줄에 JSON, 종류는 `type` 필드로 구분):

| `type` | 언제 | 내용 |
|---|---|---|
| `token` | FinalAnswerAgent/FinalGeneralAgent 토큰 | `{agent, text}` — 화면에 흐르는 최종 답변 |
| `node_enter` | 그래프 노드 진입 | `{agent, node}` — 트레이스 창 |
| `tool_call` | 툴 호출 | `{agent, tool, args, result?}` |
| `agent_status` | 에이전트 상태 한 줄 | `{agent, detail}` |
| `needs_input` | **HITL 로 멈춤** | `{kind: collect_param\|confirm, prompt, field?, action, params, options?, resume_token}` |
| `usage` | 턴 종료 | 토큰/시간 집계 |
| `done` | 종료 | `{reason: complete\|interrupted\|stopped\|error}` |
| `error` | 오류 | `{message}` |

`needs_input` + `done{interrupted}` 를 받으면 프론트는 입력창/승인버튼을 띄우고,
사용자의 답을 **같은 `thread_id` 로 다시 `/chat/stream`** 에 보내면 됩니다.

### `POST /llm/api/chat/stop`

```json
{ "thread_id": "chat-001" }
```

- **실행 중**: 중단 플래그를 세워 스트림 루프를 탈출시킵니다.
- **인터럽트 대기 중**: 사실 실행 중이 아니라 멈춰 있는 상태입니다. 그냥 두면 인터럽트가 남아
  다음 질문이 "답변"으로 오인되므로, abort 센티널(`Command(resume={"aborted": True})`)로
  재개해 `abandon` 경로를 태워 깨끗이 정리합니다.

---

## 에이전트 간 연계 (needs-핸드오프)

> ActionAgent 를 Router 에 Supervisor 와 **동급**으로 붙이면 다른 에이전트와 협업이 안 됩니다.
> 그래서 Supervisor 밑 member 로 두고, 필요한 값은 **동료에게 잠깐 양보해서** 받아옵니다.

```
"로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘"

 ActionAgent : eqp_id 는 분석이 필요하다고 판단
               → action.needs = {agent:"LogAgent", fill:"eqp_id"} 기록
               → interrupt 가 아니라 '정상 종료'로 Supervisor 에 양보
 Supervisor  : needs 감지 → LogAgent 라우팅 (결정적 분기, LLM 판단 불필요)
 LogAgent    : 자기 해석 프롬프트로 분석 → 결과를 facts 에 적재
 Supervisor  : 결과 도착 → ActionAgent 재진입
 ActionAgent : 스크래치 생존 → infer_intent 스킵, param_check 부터 재개
               → facts 에서 흡수 → 그 파라미터는 사용자에게 묻지 않음 → confirm ⏸
```

핵심: **각 에이전트의 해석용 프롬프트는 그 에이전트에 남습니다.** ActionAgent 는 "누가 필요한지"만
선언하므로 복잡도가 튀지 않습니다. HITL 과 대칭 구조이기도 합니다 —
**사람에게 물으면 `interrupt()`, 동료에게 물으면 `needs`.**

가드: `MAX_HOPS` 왕복 상한. 헬퍼가 값을 못 찾으면 **HITL 로 강등**해 사용자에게 직접 묻습니다.

---

## 액션 도중 탈출

모든 interrupt 지점에서 빠져나올 수 있습니다.

- **대화로**: resume 답변을 값/승인으로 해석하기 **전에** 취소 의도("취소/그만/됐어")를 먼저 검사 → `abandon`
- **버튼/stop**: `/chat/stop` 이 abort 센티널로 재개 → `abandon`

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
`thinking` 이벤트가 추가로 흐릅니다. 다만 interrupt 하는 노드에서 토큰을 흘리면 **재개 시 중복
방출**되므로 ActionAgent 는 interrupt 위쪽에서 스트리밍하지 않도록 짜여 있습니다.

---

## 모듈 구조

```
app/
├── config.py            # .env 로드
├── _state.py            # AgentState + 4턴 limiter + ActionScratch
├── _llm.py              # LLM 팩토리 (OpenAI 호환 endpoint / FAKE_LLM 목업)
├── _agent.py            # router/supervisor/final 등 에이전트
├── _node.py             # 노드들 + Supervisor 의 needs 라우팅
├── _util.py             # emit(SSE 커스텀 이벤트) 등
├── _builder.py          # build_team_graph — 기존 배선 + ActionAgent 삽입
├── main.py              # FastAPI 엔트리
├── actions/             # ★ ActionAgent 도메인
│   ├── mock_db.py       #   목업 DB + 공유 조회 함수(LocationAgent 도 재사용)
│   ├── registry.py      #   ActionSpec + ACTION_REGISTRY ← 액션 추가 지점
│   ├── resolvers.py     #   답변 해석 4분기 / 승인 판정
│   ├── tools.py         #   param_check / validate / confirm / execute
│   └── graph.py         #   HITL 서브그래프
└── api/
    ├── routes.py        # /chat/stream, /chat/stop, 일별 jsonl 로그
    ├── sse.py           # SSE 직렬화
    ├── usage_store.py   # thread_id 별 토큰/시간 원장
    ├── graph_service.py # 그래프 빌드 캐시
    └── schemas.py
```

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

## ActionAgent 구현 방식 두 가지 (브랜치 비교)

| | `claude/hitl-langgraph-chatbot-e67urt` | `...-singlenode` (이 브랜치) |
|---|---|---|
| 형태 | 서브그래프 — 노드/엣지로 분리 | **단일 노드 + 내부 while 루프** |
| interrupt 배치 | 노드당 정확히 1개 (`collect_param`, `confirm`) | 한 노드 안에 여러 개 (순서 매칭 의존) |
| resume 재실행 | 작은 노드만 재실행 | **노드 전체 재실행** |
| **토큰 비용(실측)** | **163 tok** | **261 tok (+60%)** |
| 흐름 가독성 | 엣지가 곧 순서도 | 코드를 읽어야 순서를 앎 |
| 코드량 | `graph.py` 398줄 (노드 11개) | `graph.py` 약 290줄 (함수 1개) |

두 브랜치는 **`app/actions/graph.py` 단 한 파일만** 다릅니다. 나머지는 완전히 동일하고
테스트(시나리오 12종 + SSE 8종)도 양쪽 다 통과하므로, **두 브랜치의 diff 가 곧 두 방식의 차이**입니다.

### 토큰 차이가 나는 이유

동일한 흐름("6PDMQ283 반송해줘" → 목적지 입력 → 승인, `/chat/stream` 3회)을 돌렸을 때
측정한 값입니다. 단일 노드 버전은 **resume 할 때마다 노드 전체가 맨 위부터 재실행**되므로
루프 위쪽의 `infer_intent` LLM 호출이 3번 반복됩니다.

```
서브그래프  : Router 21 + Supervisor 23 + infer_intent 49 + FinalAnswer 70  = 163
단일 노드   : Router 21 + Supervisor 23 + ActionAgent 147   + FinalAnswer 70  = 261
                                          └ infer_intent × 3회 (재실행 이중 과금)
```

서브그래프 버전은 `infer_intent` 를 별도 노드로 분리해, 완료된 노드는 재실행되지 않게 만들어
이 비용을 구조적으로 피합니다. HITL 왕복이 많아질수록 격차가 커집니다.

**권장: 서브그래프 버전.** 단일 노드 버전은 흐름이 한 함수에 모여 있어 읽기 쉽다는 장점이 있어
비교용으로 남겨둡니다.

---

## 사내 반입 체크리스트

1. `.env` 에 사내 값 채우기 — **변수 이름을 `pptx-vision-rag` 와 동일하게 맞춰뒀으니
   기존 `.env` 의 게이트웨이 설정을 그대로 복사**하면 됩니다.
   ```bash
   FAKE_LLM=0
   LLM_GATEWAY_BASE_URL=http://hcp.llm.skhynix.com/v1
   LLM_GATEWAY_API_KEY=            # 사내 게이트웨이는 키 불필요 → 비워둠
   LLM_CHAT_MODEL=glm-5.1
   LOG_DIR=./devLogs
   ```
   (구 이름 `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`MODEL_NAME` 도 폴백으로 인식합니다.)
2. `app/_llm.py` 의 `get_llm()` 확인 — 이미 사내 게이트웨이 호출 패턴
   (`ChatOpenAI(base_url=…, api_key=… or "EMPTY", max_tokens, max_retries, timeout)`)에
   맞춰져 있습니다. 사내 공용 래퍼(`_llm.llm_t1`)가 따로 있으면 그것으로 교체하세요.
3. `app/actions/mock_db.py` 를 실제 DB 조회로 교체 (함수 시그니처는 그대로 두면 나머지는 무수정)
4. `app/_node.py` 의 Location/Status/Log/Extract 스텁을 사내 실제 노드로 교체
   — **`_serve_needs()` 호출만 유지**하면 needs-핸드오프가 그대로 동작합니다
5. `requirements.txt` 의 langgraph/langchain-core 버전을 사내 버전에 맞추기 (아래 참고)

### 버전 (사내 확인 완료)

사내 실제 버전과 개발·검증 환경이 **동일**합니다. 별도 조정이 필요 없습니다.

| 패키지 | 사내 | 검증 |
|---|---|---|
| `langgraph` | 1.1.2 | 1.1.2 ✅ |
| `langchain-core` | 1.4.9 | 1.4.9 ✅ |
| `langchain-openai` | 1.1.8 | 1.1.8 ✅ |
| `pydantic` | 2.12.5 | 2.12.5 ✅ |

`requirements.txt` 는 위 4개를 **정확히 고정(`==`)** 합니다. LangGraph 는
`interrupt()`/`Command`/`StateSnapshot.interrupts` API 가 버전마다 달라, 올리면 HITL 재개가
깨질 수 있어서입니다. 나머지(fastapi/uvicorn/streamlit/httpx/python-dotenv)는 사내
`pptx-vision-rag` 와 같은 `>=` 하한 방식으로 두었습니다.

> 이 환경에서 `pip install -r requirements.txt` 는 아무것도 바꾸지 않습니다(전부 already satisfied).

버전이 확정됐지만 방어 코드는 그대로 둡니다 — 인터럽트 감지는 `StateSnapshot.interrupts` 를
먼저 보고 없으면 `tasks[].interrupts` 로 폴백하므로, 나중에 사내가 구버전으로 내려가도 동작합니다.
