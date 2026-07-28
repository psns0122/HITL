# 사내 코드 이식 가이드

이 저장소(`hitl_merge`)를 사내 프로젝트에 끼워 넣는 순서입니다.
사내 구조(shared_code.md §0) 기준으로, **한 단계 끝날 때마다 확인**하고 넘어갑니다.

```
사내 app/                          이 저장소 app/
├── main.py                        ├── api/main.py        (lifespan 추가분만 참고)
├── config.py                      ├── config.py          (병합)
├── _state.py                      ├── _state.py          (병합 — 필드 추가)
├── _builder.py                    ├── _builder.py        (병합 — 3줄 수준)
├── _node.py                       ├── _node.py           (병합 — Router/Supervisor)
├── _agent.py                      ├── _agent.py          (참고 — 사내 것 유지)
├── _util.py                       ├── _util.py           (헬퍼 몇 개 추가)
├── _llm.py                        ├── _llm.py            (사내 llm_t1 유지)
└── api/                           ├── api/
    ├── routes.py                  │   ├── routes.py      (병합 — 스트림 루프)
    ├── graph_service.py           │   ├── graph_service.py (교체 — 모델별 캐시)
    └── schemas.py                 │   └── schemas.py     (필드 추가)
                                   ├── actions/           ★ 통째 복사 (신규)
                                   ├── id_reader.py       ★ 통째 복사 (신규)
                                   └── _prompt.py         ★ 통째 복사 (문구는 사내화)
```

원칙: **신규 파일은 복사, 기존 파일은 최소 병합.** 사내 코드가 기준이고,
이쪽 코드는 "무엇을 더하는지"만 가져갑니다.

> **`origin/` 패키지를 먼저 보세요.** HITL 이전의 사내 원본을 스냅샷으로
> 담아둔 곳입니다. 병합 전에 `diff -u origin/<파일> app/<파일>` 을 뜨면
> "이 단계에서 정확히 무엇이 늘어나는지" 가 한눈에 보입니다.
> 파일별 근거(직접 제공 / 첨부 / 사내 규약 / 추정)는 `origin/README.md` 에
> 태그로 표시돼 있고, `[원본·추정]` 인 파일은 사내 실물로 덮어써야 합니다.

---

## 0단계 — 사전 준비

1. 사내 프로젝트에 새 브랜치를 파고 시작합니다.
2. 의존성 확인 (이미 버전 일치 확인됨):
   - `langgraph==1.1.2`, `langchain-core==1.4.9`, `langchain-openai==1.1.8`, `pydantic==2.12.5`
   - ⚠️ `langgraph-prebuilt==1.0.8` — 1.0.9+ 는 langgraph 1.1.2 에서 ImportError.
     (단, 턴 기반은 create_react_agent 를 안 쓰면 prebuilt 자체가 불필요할 수 있음)
3. `.env` 는 그대로 씁니다 — 변수명이 `LLM_GATEWAY_*` 로 이미 사내 규약.

**확인**: `python -c "import langgraph, langchain_core; print('ok')"`

---

## 1단계 — 신규 파일 통째 복사 (기존 코드와 충돌 없음)

| 파일 | 내용 | 사내 반입 시 손볼 곳 |
|---|---|---|
| `_action.py`/`_db.py` + `_tool.py`·`_prompt.py` 의 ActionAgent 섹션 | ActionAgent 도메인 (선언은 `_prompt.action_catalog()`) | `_db.py` → 나중에 실 DB (7단계) |
| ├ `node.py` | **턴 기반 HITL 단일 노드** (핵심) | 없음 |
| ├ `registry.py` | ActionSpec + 액션 2종 등록 | 액션 추가 시 여기만 |
| ├ `tools.py` | param_check / validate / confirm / execute | validate/execute 본문 (7단계) |
| ├ `resolvers.py` | 답변 판정 (취소/값/상담/맥락이탈) | 취소·전환 키워드 사내 어휘 보강 |
| └ `_db.py` | 목업 DB + 공유 조회 함수 | 실 DB 로 교체 (7단계) |
| `app/id_reader.py` | ID 판독기 (정규식 후보 → 존재 조회) | `id_lookup_tool` 본문 → 실 DB (7단계) |
| `app/_prompt.py` | 프롬프트 모음 | 문구 전부 사내화 가능 (로직 무관) |
| `app/api/usage_store.py` | thread별 토큰/시간 원장 | 선택 (안 쓰면 routes 에서 호출 제거) |
| `app/api/limits.py` | 동시성/유량 제어 | 선택 |

**확인**: `python -c "from app._action import action_node; print('ok')"`

---

## 2단계 — `_state.py` 병합

사내 `AgentState` (messages limiter/route/handoff/next/step) 는 **그대로 두고**
필드만 추가합니다:

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], create_limiter(max_turns=4)]  # 기존
    route: Literal["general", "supervisor"]                              # 기존
    handoff: bool                                                        # 기존
    next: str                                                            # 기존
    step: int                                                            # 기존
    model_name: str                          # ★ 추가 — 프론트 선택 모델
    action: ActionScratch                    # ★ 추가 — annotation 없음(교체)
    facts: Annotated[dict, merge_dict]       # ★ 추가 — 병합 리듀서
```

+ 이 저장소 `_state.py` 의 `ActionScratch`, `merge_dict` 를 함께 복사.

핵심: `action` 에 리듀서를 달지 마세요(last-write-wins 여야 함).
HITL 진행 상태를 messages 에 두면 limiter 에 잘려서 깨집니다 — 그래서 밖에 둡니다.

**확인**: import 만 되면 됨.

---

## 3단계 — Router 이식 ★ 가장 많이 바뀐 곳, 여기부터 차근차근

사내 `router_node` 대비 바뀐 것은 딱 **두 가지**입니다.

**(a) 최상단에 "진행 중 액션 → Supervisor 고정" 4줄 추가** — 이게 턴 기반 HITL 의 심장:

```python
def router_node(state, config):
    ...기존 프린트...
    # ★ 추가: 진행 중 액션이 있으면 분류 없이 Supervisor 고정.
    #   HITL 답변("STK102")이 general 로 오분류되는 것을 원천 차단한다.
    if (state.get("action") or {}).get("phase"):
        return {"route": "supervisor", "handoff": False, "next": "Supervisor", "step": 1}
    ...기존 분류 로직 그대로...
```

**(b)** (선택) 분류를 `classify_route_with_llm` 방식(JSON 강제 + normalize_route_label
폴백)으로 교체 — 사내 `router_agent` 가 이미 잘 돌면 안 바꿔도 됩니다.
async 전환도 선택입니다 (LangGraph 는 sync 노드도 받음).

**확인 — 이 3개가 전부 통과해야 다음 단계로**:

```python
# 1) 일반 질의 -> GeneralAgent 로 가는가
await graph.ainvoke({"messages": [HumanMessage("안녕!")]}, cfg)
#    콘솔: Router -> GeneralAgent

# 2) 업무 질의 -> Supervisor 로 가는가
await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 위치")]}, cfg)
#    콘솔: Router -> Supervisor

# 3) ★ awaiting 상태에서 아무 답변이나 -> Supervisor 고정인가
#    (액션 진행 중 스레드에서) "STK102" 입력
#    콘솔: "Router: 진행 중 액션 감지 -> Supervisor 고정" 이 찍혀야 함
```

3)이 안 되면 이후 전부 무너집니다. 반드시 확인하고 넘어가세요.

---

## 4단계 — Supervisor 이식

사내 `supervisor_node` 의 **LLM 배분(supervisor_chain) 앞에** 결정적 우선순위
블록을 삽입합니다. 이 저장소 `_node.py` 의 supervisor_node 에서 0)~5) 블록을
그대로 가져오면 됩니다:

```
0) needs 상담      : ActionAgent 의 상담 요청 -> needs_dispatch 로 워커 배분
0-b) 상담 회수     : 워커 답변 원문을 메일박스에 실어 ActionAgent 반송
1) 헬퍼 결과 도착   : ActionAgent 재진입
1.5) awaiting 턴 닫기: ActionAgent 가 질문을 던졌으면 -> next="END" (FinalAnswer 없이)
2) 진행 중 액션    : ActionAgent 계속
3) Extract 선행    : 이번 턴에 안 돌았으면 무조건 먼저
3-b) Extract 게이트: 추출 결과 없으면 바로 FinalAnswer
4) 워커 응답 완료   : FinalAnswerAgent (ANSWERING_MEMBERS — Extract 제외!)
5) 스텝 상한 가드
--- 여기부터 기존 사내 LLM 배분 그대로 ---
```

+ `_builder.py` 의 supervisor_conditional_map 에 한 줄 추가:

```python
supervisor_conditional_map["END"] = END   # awaiting 턴은 FinalAnswer 없이 종료
```

+ `_agent.needs_dispatch` 와 `_prompt.needs_dispatch_prompt` 복사 (상담 배분용).
+ `_util.py` 에 `member_answered_this_turn` / `agent_ran_this_turn` 복사.

주의 2가지:
- `ANSWERING_MEMBERS` 에서 ExtractAgent 를 빼야 합니다. 안 빼면 Extract 가
  돌자마자 "워커가 답했다"로 보고 턴이 끝납니다.
- Extract 선행이 사내에 필요 없으면 3)/3-b) 는 빼도 됩니다 (독립 블록).

**확인**: "6PDMQ283 반송해줘" 한 방에 콘솔 노드 순서가
`Router → Supervisor → ExtractAgent → Supervisor → ActionAgent → Supervisor(END)`
이고, FinalAnswer 없이 끝나며, state 의 `action.awaiting` 에 질문이 실려 있는지.

---

## 5단계 — `_builder.py` + `graph_service.py`

`_builder.py` 변경은 사실상 3줄입니다:

```python
from app._action import build_action_node
workflow.add_node("ActionAgent", build_action_node())   # 기존 action_node 자리에
supervisor_conditional_map["END"] = END                  # 4단계에서 이미
```

members 리스트에 "ActionAgent" 가 있고 복귀 엣지(`for member in members:
add_edge(member, "Supervisor")`)가 도는 건 사내 코드 그대로면 자동입니다.

⚠️ **체크포인터는 모델별로 나누면 안 됩니다.** 그래프는 모델별 캐시라도
체크포인터는 프로세스 공용 하나(SHARED_CHECKPOINTER)를 공유해야, 승인 대기 중
모델을 바꿔도 HITL 이 이어집니다. `graph_service.py` 는 이 저장소 것으로 교체
(모델명 키 캐시 + 공용 체크포인터).

**확인**: `tests/test_hitl_scenarios.py` 의 A(수집→승인→실행)/B(거절) 시나리오를
사내 그래프로 돌려 통과.

---

## 6단계 — `api/routes.py` + `schemas.py`

`schemas.py`: ChatRequest 에 `model_name: str | None = None`,
`recursion_limit: int = Field(20, ge=1, le=200)` 추가.

`routes.py` 는 사내 스트리밍 루프를 유지한 채 **세 가지만** 더합니다:

1. **입력은 항상 새 턴** — 턴 기반이라 resume 분기가 아예 없습니다.
   기존 사내 코드(`{"messages":[HumanMessage(q)]}` 입력)가 그대로 정답.
   `model_name` 만 inputs 와 config.configurable 에 실어 주세요.
2. **스트림 종료 후 HITL 감지** — astream_events 루프가 끝나면:
   ```python
   snap = await graph.aget_state(config)
   awaiting = (snap.values.get("action") or {}).get("awaiting")
   if awaiting:  # needs_input + usage 프레임 송출, 로그는 아직 안 씀
   ```
3. **/chat/stop** — HITL 대기는 실행 중이 아니므로
   `await graph.aupdate_state(config, {"action": {}})` 로 스크래치만 리셋.

제어 프레임(`\x1e` JSON)은 사내 raw-text 스트림 위에 얹는 방식이라
기존 토큰 송출부는 안 바꿔도 됩니다. 프론트 파서는 `streamlit_app.py` 의
`stream_chat()` 을 복붙.

**확인** (서버 띄우고 curl 3연타):
```
질의  "6PDMQ283 반송해줘"  -> \x1e{"type":"needs_input","kind":"collect_param",...}
답변  "STK102"             -> \x1e{...,"kind":"confirm",...}
승인  "승인"               -> "✅ ... TJ-..." raw text + usage + done{complete}
로그  logs/{env}/{월}/{일}.jsonl 에 1건 (스트림 3회가 합산됐는지)
```

---

## 7단계 — 목업 → 실물 교체 (한 곳씩, 각각 테스트)

| 교체 지점 | 파일 | 비고 |
|---|---|---|
| ID 판독기 | `id_reader.py` 의 `id_lookup_tool` 본문 | 후보 문자열 → DB 존재 조회. **여기 한 곳만** 바꾸면 수집/상담/추출 전부 따라옴 |
| 검증/실행 | `_tool.py (ActionAgent 섹션)` 의 `*_validate_tool` / `*_execute_tool` | 시그니처 유지하면 node.py 무수정 |
| LLM | `_llm.get_llm` | 이미 게이트웨이 규약(placeholder EMPTY 등) 맞춰둠. 사내 llm_t1 이 있으면 그걸로 |
| 워커 스텁 | `_node.py` 의 location/status/log/extract | 사내 실제 노드로 교체. **계약 하나만 유지**: 결과를 `AIMessage(name=에이전트명)` 으로 messages 에 남길 것 (Supervisor 의 상담 회수가 그걸 읽음) |
| 프롬프트 | `_prompt.py` | 문구 교체 자유 |

**확인**: 교체 하나마다 `tests/test_hitl_scenarios.py` 재실행.

---

## 이식하지 않는 것

- `_db.py` 의 데이터 (교체 대상)
- `streamlit_app.py` — 사내 프론트가 따로 있으면 `stream_chat()` 파서만 가져감
- `notebooks/`, `docs/`, `tests/` — 원하는 만큼만

## 순서 요약

```
0 준비 → 1 신규복사 → 2 _state → 3 Router(★검증 3종) → 4 Supervisor
→ 5 _builder/graph_service → 6 routes/schemas → 7 목업→실물 (하나씩)
```

각 단계 사이에 커밋해 두면, 어디서 깨졌는지 바로 좁혀집니다.
