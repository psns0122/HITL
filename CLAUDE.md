# 이 저장소에서 작업하는 규칙

## 대전제: 모든 app 코드는 결국 origin(사내 코드)으로 이식된다

```
origin/   사내 원본 코드의 스냅샷. 이식의 기준선. 여기가 "진실"이다.
app/      origin + HITL ActionAgent. 이 저장소에서 실제로 도는 코드.
```

**app 을 수정할 때는 항상 "이 diff 를 사내 코드에 그대로 얹을 수 있는가"를
먼저 생각한다.** 목적을 달성하는 여러 방법이 있으면, origin 과의 diff 가
가장 작아지는 방법을 고른다. 최소 수정으로 목적 달성 — 그게 기준이다.

## app 수정 규칙

1. **origin 에 있는 함수/파일은 시그니처·반환 형태·이름을 origin 과 맞춘다.**
   같은 일을 하는 코드가 양쪽에서 다르게 생기면 이식 때 병합 지옥이 된다.
2. **새 기능은 가능하면 기존 파일의 새 섹션/새 클래스로 붙인다** (예: `app/_util.py 의 ActionService`).
   사내에 이미 같은 일을 하는 함수가 있으면 **새 파일을 만들지 말고 그것을 쓴다**
   (예: ID 판독 → `params_extract_tool`. 별도 판독기 모듈을 두면 이식 때 버려진다).
   기존 파일 수정은 "몇 줄 추가" 수준으로 유지한다 — 통째 재작성 금지.
3. **origin 과 일부러 다르게 가는 부분은 그 파일에 주석으로 이유를 남긴다.**
   이유를 못 쓰겠으면 다르게 갈 이유가 없는 것이다.
4. origin ↔ app 은 서로 import 하지 않는다.
5. 수정 후 `diff -u origin/<파일> app/<파일>` 로 diff 가 의도한 만큼만
   커진 것을 확인한다.

## origin 수정 규칙

- origin 은 **사용자가 타이핑/첨부해 준 사내 코드만** 들어간다.
  임의로 함수를 만들거나 지우지 않는다. 필요한 게 비면 사용자에게 목록으로 묻는다.
- 타이핑 오타(변수명 오타, 괄호 누락 등)는 고치되 구조·로직은 손대지 않는다.
- 원문이 없어 채워 넣은 부분은 `[원본·추정]` 태그를 달고 사내 실물로 덮어쓸
  대상임을 명시한다. (태그 체계: `origin/README.md`)

## 프로젝트 불변식 (사내 규칙 — 어기지 말 것)

- **판단은 전부 LLM 이 한다.** 룰/정규식 기반 판단 금지. 목업 LLM 금지.
  (예외: ID 인식은 판독기 툴의 DB 조회 — LLM 판단이 아니다)
- 모든 사용자 입력은 Router → Supervisor 를 경유한다. HITL 답변도 예외 없다.
  (turn 기반 HITL — LangGraph `interrupt()` 를 쓰지 않는다)
- 실행(execute)은 명시적 승인(approve) 이후에만 도달한다.
- 설정값은 코드가 아니라 `.env` 로. API 키를 코드에 두지 않는다.
- 로깅은 별도 라이브러리 없이 `print(..., flush=True)`. 공용 `_log` 헬퍼 금지 —
  각 함수가 직접 찍는다.
- 모든 툴/노드는 진입 시점과 진행 과정을 print 로 남긴다.

## 현재 알려진 origin ↔ app 구조 차이 (이식 시 병합 지점)

| 항목 | origin | app | 비고 |
|---|---|---|---|
| ActionAgent | 일반 react agent | 턴 기반 HITL **3단 파이프라인** (`_node.py`: ActionAgent 판단 → ActionValidator 검증·승인질문 → ActionExecutor 판정·실행. 흐름은 flow_tool/decide_tool 로 에이전트가 선언) | **이번 작업의 본체** — 노드 3개 + 빌더 배선 + Supervisor 배관을 함께 이식. 액션 선언은 `_prompt.action_catalog()`, 툴 바인딩은 네이밍 규칙 |
| Supervisor | LLM 배분만 | + needs-핸드오프 / Extract 선행 / HITL 턴 종료 우선순위 | 병합 필요 |
| AgentState | messages/route/handoff/next/step/model_name | + `action`, `facts` | 필드 추가 |
| Router 의 next | 노드 이름("Supervisor"/"GeneralAgent") | 동일 (수렴 완료) | — |
| 에이전트 이름 표기 | `additional_kwargs["agent_name"]` | 동일 + `name=` 폴백 병기 (수렴 완료) | 판독은 `_util.agent_name_of` |
| model_name 전달 | state + `config.configurable` 둘 다 | 동일 (수렴 완료) | — |
| 스트림 | raw text | SSE 5 이벤트 (`docs/API.md`) | app 확장 — 이식 시 routes 병합 |
| ID 판독 | `params_extract_tool` (DB) | 동일 — `_tool.py 의 params_extract_tool` (수렴 완료) | ExtractAgent·ActionAgent 가 같은 툴을 쓴다. 이식 시 본문은 사내 것 유지 |

이식 절차 자체는 `docs/PORTING.md`.
