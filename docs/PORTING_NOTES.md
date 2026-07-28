# 사내 이식 시 신경 써야 하는 것들 (hitl_merge → 사내 origin)

> 단계별 **절차**는 `PORTING.md` 에 있다. 이 문서는 그 절차를 밟는 동안
> **무엇을 조심해야 하는가** — 병합 지점, 깨지기 쉬운 불변식, LLM 판단 지도,
> 그리고 최근 변경분(2026-07-29)의 이식 포인트를 정리한 것이다.

---

## 1. 큰 그림

```
origin/   사내 원본 스냅샷 = 이식의 기준선. 여기가 "진실".
app/      origin + HITL ActionAgent. diff 가 곧 이식할 내용.
```

이식은 "app 을 들고 가는" 작업이 아니라 **"origin 대비 diff 를 사내 코드에
얹는" 작업**이다. 시작 전에 반드시:

```bash
diff -u origin/_node.py  app/_node.py
diff -u origin/_agent.py app/_agent.py
diff -u origin/_util.py  app/_util.py
diff -u origin/_llm.py   app/_llm.py
diff -u origin/config.py app/config.py
diff -u origin/_state.py app/_state.py
```

`origin/README.md` 의 태그를 확인할 것 — `[원본·추정]` 파일은 사내 실물과
다를 수 있으므로, **diff 를 뜨기 전에 사내 실물로 덮어쓰고 다시 diff** 한다.

---

## 2. 파일별 병합 지도

### 그대로 얹으면 되는 것 (app 전용 — 사내에 대응물 없음)

| 대상 | 위치 | 주의 |
|---|---|---|
| `ActionService` (턴 기반 HITL 본체) | `_util.py` 하단 `[app 전용]` 블록 | **통째 복사.** 상단 origin 영역과 섞이지 않게 블록 경계 주석 유지 |
| ActionAgent 판단부 (4 함수 + `_ActionAgent`) | `_agent.py` 하단 | `extract_intent` / `classify_collect_answer` / `classify_confirm` / `answer_seems_informative` |
| `needs_dispatch` + `DispatchOut` | `_agent.py` | Supervisor 의 상담 배분 |
| action 계열 프롬프트 5종 | `_prompt.py` | `action_catalog` / `action_select` / `action_collect_answer` / `action_informative` / `action_confirm` + `needs_dispatch_prompt`. 문구 사내화 자유(로직 무관) |
| ActionAgent 툴 섹션 | `_tool.py` | `param_check` / `*_validate` / `*_confirm` / `*_execute`. **네이밍 규칙(`{action}_*_tool`)이 바인딩이다** — 이름 바꾸면 `getattr` 호출이 깨진다 |
| `ActionScratch`, `merge_dict` | `_state.py` | |
| `usage_store.py`, SSE 스트림 확장 | `api/` | 선택 — 안 가져가면 routes 에서 호출부만 걷어낸다 |

### 구조화 출력 관례 — origin 을 따른다 (2026-07-29 정리)

origin 전체에서 구조화 출력을 쓰는 곳은 **`_node.RouteResponse` 딱 하나**다.

```python
# origin/_node.py — 이게 이 프로젝트의 유일한 선례이자 기준
class RouteResponse(TypedDict):
    next: Annotated[Literal[tuple(options_for_next)], "다음에 실행할 노드"]
llm.with_structured_output(RouteResponse, method="json_mode")
```

app 은 이 관례를 따른다. **스키마는 pydantic BaseModel 이 아니라 TypedDict**,
정의 위치는 그 값을 쓰는 파일, 값 제약은 `Literal` 로 스키마에 박는다.

한 곳만 의도적으로 다르다 — **app 의 ActionAgent 판단부는 `json_mode` 를 쓰지 않는다.**

| | 대상 | method | 이유 |
|---|---|---|---|
| Supervisor 배분 | `_node.RouteResponse` (필드 1개) | `json_mode` (origin 그대로) | 프롬프트 본문의 [Strict Output Rule] 로 충분 |
| ActionAgent 판단 4종 | `_agent.py` 의 TypedDict | 기본(스키마 전달) | `json_mode` 는 **스키마를 모델에 보내지 않는다**. `IntentOut` 은 필드가 6개고 "옮길 대상 캐리어 vs 위치 기준 캐리어" 처럼 헷갈리는 짝이 있어, 스키마 없이 두면 두 ID 가 서로 뒤집힌다(실측 확인) |

이 차이는 `_agent.py` 의 `[app 전용]` 블록 머리에 주석으로 적혀 있다.

### 사내 코드에 몇 줄 병합하는 것 (충돌 주의 지점)

| 파일 | 병합 내용 | 왜 조심해야 하나 |
|---|---|---|
| `_state.py` | `AgentState` 에 `model_name`, `action`, `facts` 3필드 추가 | `action` 에 **리듀서를 달면 안 된다**(last-write-wins). `facts` 는 `merge_dict` 리듀서. HITL 상태를 messages 에 넣으면 limiter 에 잘려 깨진다 |
| `_node.py` Router | 최상단 "진행 중 액션 → Supervisor 고정" 4줄 | **턴 기반 HITL 의 심장.** 이거 없으면 HITL 답변("STK102")이 general 로 새서 액션이 고아가 된다 |
| `_node.py` Supervisor | LLM 배분 **앞에** 결정 블록 0)~5) 삽입 | 우선순위 순서 자체가 로직이다: needs 상담 → 회수 → awaiting 턴닫기 → 진행중 액션 → Extract 선행 → member 답변 완료. `ANSWERING_MEMBERS` 에서 ExtractAgent 빼는 것 잊으면 Extract 직후 턴이 끝나버린다 |
| `_node.py` action_node | 사내 react agent 노드를 `action_service.action_node` 위임 한 줄로 교체 | 워커 스텁(location/status/log/extract)은 가져가지 **않는다** — 사내 실물 유지. 단 결과를 `AIMessage(additional_kwargs={"agent_name": ...})` 로 남기는 계약만 확인 |
| `_builder.py` | `supervisor_conditional_map["END"] = END` 한 줄 | awaiting 턴을 FinalAnswer 없이 닫는 경로 |
| `_llm.py` | `structured_invoke` 는 **필수 반입** | 판단 전부가 이걸 탄다. `[app 전용]` 블록 중 DEFAULT_MODEL 덮어쓰기/AVAILABLE_MODELS 는 로컬 실행용 — 사내에선 .env 비워두면 무동작이므로 들고 가도 무해, 들어내도 된다 |
| `config.py` | `MAX_COLLECT/MAX_VALIDATE/MAX_HOPS/CONFIRM_TTL_SEC` | 전부 .env 기반, 기본값 내장 |
| `api/schemas.py` | `model_name`, `recursion_limit` 필드 추가 | |
| `api/routes.py` | ① 입력은 항상 새 턴(resume 분기 없음) ② 스트림 종료 후 `action.awaiting` 보고 needs_input 프레임 ③ /chat/stop 은 스크래치 리셋 | 사내 토큰 송출부는 건드리지 않는다 |
| `api/graph_service.py` | 모델별 그래프 캐시 + **공용 체크포인터 1개** | 체크포인터를 모델별로 나누면 승인 대기 중 모델 바꿀 때 HITL 이 끊긴다 |

### 이식하지 않는 것

- `_db.py` (목업), 워커 스텁 본문, `streamlit_app.py`(파서 `stream_chat()` 만 참고),
  `notebooks/`, `tests/`(원하는 만큼), `_llm.py 의 경고 필터`(선택)

---

## 3. 어기면 안 되는 불변식 — 이식 중 실수하기 쉬운 순서로

1. **모든 사용자 입력은 Router → Supervisor 를 경유한다. HITL 답변도.**
   `interrupt()`/`Command(resume=...)` 를 쓰지 않는다. 사내에 interrupt 기반
   코드가 남아 있다면 섞지 말 것 — 두 방식이 공존하면 재실행(replay) 함정이 돌아온다.
2. **execute 는 명시적 approve 이후에만.** `_execute_and_finalize` 가 유일한
   side-effect 지점이다. 이식 중 "편의상" 다른 경로에서 execute 를 부르지 않는다.
3. **서브그래프를 추가하지 않는다.** ActionAgent 는 단일 노드(`ActionService`)다.
   턴 기반은 interrupt 재실행 격리가 필요 없으므로 쪼갤 이유가 없다.
4. **판단은 전부 LLM.** 룰/정규식 판단 금지. 예외 두 가지만: ID 판독(판독기 툴의
   DB 조회)과 승인 TTL(시계 비교 — 판단이 아니라 시간 초과).
   LLM 실패 시 폴백은 항상 안전한 쪽: 재질문 / 미승인 / FinalAnswer.
5. **체크포인터는 프로세스 공용 하나.** 그래프 캐시는 모델별이어도 된다.
6. **`action` 스크래치는 리듀서 없이, messages 밖에.** (limiter 에 잘리면 HITL 파탄)
7. 로깅은 `print(..., flush=True)` 직접. 공용 `_log` 헬퍼 만들지 않는다.
8. 설정은 `.env` 로. 키를 코드에 두지 않는다.

---

## 4. LLM 이 직접 판단하는 지점 (전체 지도)

"판단은 전부 LLM" 규칙의 실제 구현 위치. 이식 후 이 표의 함수가 전부
`structured_invoke` 또는 스트리밍 호출로 실 LLM 을 타는지 확인할 것.

| # | 주체 | 함수 (위치) | 판단 내용 | 실패 시 폴백 |
|---|---|---|---|---|
| 1 | Router | `classify_route_with_llm` (`_agent.py`) | general / supervisor 분류 | supervisor (과잉 조회가 낫다) |
| 2 | GeneralAgent | 같은 함수, temperature 0.5 재판정 + 일반 답변 생성 | 업무 질의 세이프티 핸드오프 | — |
| 3 | Supervisor | `supervisor_agent` (`_agent.py`) | 워커 배분 (구조화 출력) | FinalAnswerAgent |
| 4 | Supervisor | `needs_dispatch` (`_agent.py`) | 상담 요청을 풀 헬퍼 워커 + 질의문 선정 | NONE → 사용자에게 직접 질문 |
| 5 | ActionAgent | `extract_intent` (`_agent.py`) | 최초 발화 → 액션/파라미터/참조/취소 | 빈 결과 → 사용자에게 물어봄 |
| 6 | ActionAgent | `classify_collect_answer` (`_agent.py`) | 파라미터 질문의 답 분류 (cancel/consult/switch/action/value/empty) | empty → 재질문 |
| 7 | ActionAgent | `classify_confirm` (`_agent.py`) | 승인 판정 (approve/reject/unclear) | unclear → **절대 승인 안 됨** |
| 8 | ActionAgent | `answer_seems_informative` (`_agent.py`) | 판독 실패 답변의 상담 가치 | False → 재질문 |
| 9 | 워커 4종 | `create_*_agent` react 루프 (`_agent.py`) | 툴 선택/호출 (사내 실물 기준) | — |
| 10 | Final* | `_make_final_agent` (`_agent.py`) | 최종 답변 생성 (스트리밍, 재시도 1회) | 폴백 문구 |

LLM 이 **아닌** 것: `params_extract_tool`(ID 후보 추출 + DB 존재 판정),
`param_check_tool`(필수 파라미터 대조), `*_validate_tool`(도메인 검증),
승인 TTL(시계). 이들은 판단이 아니라 조회/대조다.

---

## 5. 2026-07-29 변경분 — 이식에 포함할 것

이번 로컬 검증에서 넣은 변경. 전부 action-HITL 영역이라 origin diff 는 늘지 않는다.

| 변경 | 파일 | 성격 |
|---|---|---|
| **승인 TTL** — `confirm` 발행 시 `asked_at` 기록, `CONFIRM_TTL_SEC`(기본 60s) 초과 답변은 내용 무관 만료 → abandon | `config.py`, `_util.py` (`_ask_confirm` + 진입부), `.env.example` | 신규 |
| **confirm 답변 처리 순서 교정** — 의도 분류(cancel/switch)를 **파라미터 정정보다 먼저**. ID 실린 새 질문("6PDMQ283 위치 찾아줘")을 정정으로 오인해 캐리어를 갈아끼우던 버그 수정 | `_util.py` `_consume_confirm_answer` | 버그 수정 |
| **`answer_seems_informative` 신설** — 기존 코드가 미정의 함수를 호출하고 있었다(AttributeError 잠복). LLM 판정으로 구현 | `_agent.py`, `_prompt.py` | 버그 수정 |
| **collect 분류 프롬프트 보강** — switch 예시("로그 분석 해줘" 등)와 switch↔value 구분 규칙 추가. 소형 모델이 새 요청을 value 로 오분류해 탈출 못 하던 문제 | `_prompt.py` | 프롬프트 |
| TTFT 사람 대기 제외 — HITL 라운드 동안 사용자가 고민한 시간을 "첫 응답" 지표에서 뺀다 | `api/usage_store.py` | app 전용(선택) |
| HITL 대기 트레이스 표기 `▶ Agent` + `- 사용자 입력 대기` | `streamlit_app.py` | 이식 대상 아님 |

### origin 회귀 정리 (같은 날, 별건)

"HITL 과 무관한 곳은 origin 을 베낀다" 규칙에 맞게 되돌린 것들. **이식 부담이
줄어드는 방향이므로 반드시 함께 반영한다.**

| 되돌린 것 | 무엇을 했는가 |
|---|---|
| `SupervisorOut` 삭제 | origin 에 이미 있던 `_node.RouteResponse`(TypedDict + `json_mode` + `Literal` 제약)로 복원. 같은 일을 하는 클래스를 이름·파일·타입까지 바꿔 새로 만들어 뒀던 것 |
| `_agent.supervisor_agent()` 삭제 | origin 처럼 `supervisor_node` 안에서 `supervisor_prompt \| llm.with_structured_output(...)` 체인을 직접 만들고 3단계 방어까지 origin 그대로. 모듈 레벨 `safe_supervisor_prompt`/`supervisor_prompt` 도 복원 |
| `supervisor_agent_prompt(members)` → `()` | origin 시그니처 복원. origin 주석이 "인자를 받지 않고 본문에 중괄호를 쓰지 않는다"고 명시했는데 app 이 f-string 으로 바꿔 놨던 것 |
| `IntentResult` dataclass 삭제 | `IntentOut`(스키마) → dataclass 재변환 이중 구조를 없애고 정규화된 dict 하나만 반환 |
| `InformativeOut` + `answer_seems_informative` 삭제 | 같은 답변을 두 번 LLM 에 묻던 구조. 이 지점에 오면 이미 `classify_collect_answer` 가 value 로 분류한 답이므로 그 판단을 신뢰한다(코드 주석이 원래 그렇게 적혀 있었다). LLM 호출 1회 감소 |
| `_llm.structured_invoke` 삭제 | origin 에 없는 헬퍼. 호출부가 origin 처럼 `with_structured_output` 을 직접 쓴다 |
| Final 섹션 복원 | `_clean_messages`/`_build_final_chain`/`_astream_final`/`_make_final_agent` 4개로 쪼개고 반환을 str 로 바꿨던 것을 origin 형태(두 함수 각자 완결, `{"messages": [...]}` 반환, `finish_reason` 검사)로 되돌림. `astream` 과 `config` 인자만 SSE 때문에 남기고 각각 주석 처리 |
| `_agent.py` 배치 | app 전용 코드가 파일 한가운데 있던 것을 하단 `***** [app 전용] *****` 블록으로 이동 (`_util.py`/`_llm.py`/`config.py` 와 같은 관례). 위쪽은 origin 순서와 일치 |
| pydantic 클래스 7 → 4 | `_agent.py` 의 구조화 출력 클래스가 `DispatchOut`/`IntentOut`/`CollectAnswerOut`/`ConfirmOut` 4개만 남음. 전부 TypedDict |

추가로, origin 프롬프트를 그대로 쓰니 **작은 모델에서 Supervisor 가 0/5 로 깨졌다.**
origin 의 `supervisor_prompt` 는 `MessagesPlaceholder` 로 끝나는데, 대화의 마지막이
거의 항상 워커의 `AIMessage` 라서 모델이 담당자를 고르는 대신 그 문장을 이어 쓴다.
프롬프트 맨 끝에 사람 차례 한 줄(`("human", "위 대화 기준으로 …")`)을 더하면 5/5 가
된다 — `tools/probe_supervisor.py` 로 재현/측정할 수 있다. 사내 모델에도 무해하다.

> 참고로 origin 의 `supervisor_node` 는 반환값 `"messages"` 에 기존 messages 리스트를
> 그대로 실어 보낸다(원문 주석도 "타입이 갈리지만 원문의 모양"이라고 인정). app 은
> 이것만 따르지 않는다 — messages 리듀서가 add 라서 대화가 통째로 중복 누적된다.
> 해당 위치에 주석으로 이유를 남겨 뒀다.

검증 완료(로컬 glm4:9b): TTL 만료 → 승인 눌러도 미실행 / confirm 대기 중
"Log 분석 해줘" → LogAgent 재라우팅 / "6PDMQ283 으로 바꿔줘" → 파라미터 정정 유지.

---

## 6. 이식 후 검증 순서 (요약)

1. `PORTING.md` 3단계의 Router 검증 3종 — **3)번(awaiting 중 Supervisor 고정)이
   깨지면 이후 전부 무너진다.**
2. `tests/test_hitl_scenarios.py` — 수집→승인→실행 / 거절 / 취소 / 맥락이탈.
3. 추가된 케이스 수동 확인:
   - confirm 대기 → 60초 방치 → 승인 → `🚫 ... 승인 유효시간` 이어야 함
   - confirm 대기 → 무관한 새 질의 → 해당 워커로 재라우팅
   - confirm 대기 → "\<다른 ID\> 로 바꿔줘" → 정정 후 재승인 질문
4. 로그에 `[AGENT] ...(llm) -> ...` 가 판단마다 찍히는지 — 안 찍히는 판단이
   있으면 룰 기반이 섞여 들어간 것이다.
