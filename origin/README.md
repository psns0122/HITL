# `origin/` — 사내 원본 코드

HITL 을 붙이기 **전**의 사내 챗봇 코드입니다. 이식할 때 "원래 무엇이었는지"를
보기 위한 기준선이고, 앱은 이 패키지를 import 하지 않습니다.

```
origin/          ← 사내 원본 (이 패키지). 실행되는 코드가 아님. 비교 기준.
app/             ← 원본 + HITL ActionAgent. 실제로 도는 코드.
```

이식 순서와 병합 지점은 `docs/PORTING.md` 를 보세요.
여기서는 **diff 를 뜨는 용도**로 쓰면 됩니다.

```bash
diff -u origin/_node.py app/_node.py
diff -u origin/_builder.py app/_builder.py
```

---

## 출처 표기 (중요)

사내망에서 코드를 그대로 받을 수 없었기 때문에, 파일마다 근거가 다릅니다.
각 파일 상단에 아래 태그를 달아 뒀습니다. **태그를 보고 신뢰도를 판단하세요.**

| 태그 | 뜻 | 해당 파일 |
|---|---|---|
| `[원본·직접]` | 사용자가 직접 타이핑해 준 코드. 구조·변수명 그대로. | `_agent.py`, `_node.py` 의 `extract_node` / `supervisor_node`, `_tool.py` 의 시그니처 |
| `[원본·첨부]` | 공유해 준 `shared_code.md` 기준 재구성. | `_state.py`, `_util.py`, `_builder.py`, `api/routes.py` |
| `[원본·규약]` | 사내 다른 프로젝트(`pptx-vision-rag`)의 실제 규약에서 가져옴. | `config.py`, `_llm.py` |
| `[원본·추정]` | 위 셋 어디에도 원문이 없어 같은 패턴으로 채운 부분. **사내 실물과 대조 필요.** | `_prompt.py`, `api/schemas.py`, `api/graph_service.py`, `main.py`, `_util.safe_tool`, `_tool.py` 의 ActionAgent 툴 2종, `config.VALID_FABS` |

`[원본·추정]` 파일은 사내에서 실물을 보고 덮어써 주세요.
그래야 이 패키지가 진짜 기준선이 됩니다.

---

## 툴 계약 (`_tool.py`)

시그니처는 직접 제공받은 것이고, **본문만** 비어 있습니다
(`NotImplementedError`). 예외 없는 규약이 넷 있습니다.

```python
@tool("이름", description=_prompt.이름_description())   # ← 설명은 _prompt.py 에서
@safe_tool                                              # ← 예외를 에러 dict 로
def 이름(..., config: RunnableConfig = {}) -> Dict[str, Any]:
    """사람이 읽는 설명 (LLM 은 이걸 안 본다)"""
```

1. **툴 설명은 docstring 이 아니라 `_prompt.py` 의 `*_description()`** 에서 온다.
   → 툴이 언제 불릴지를 조정할 때 툴 코드를 건드리지 않는다.
2. 마지막 인자는 항상 `config: RunnableConfig = {}`.
   LangChain 이 자동 주입하므로 LLM 이 보는 인자 목록에는 안 들어간다.
3. 반환은 항상 `Dict[str, Any]`. 문자열이 아니다 —
   에이전트 프롬프트가 이 dict 를 해석해 리포트로 다듬는다.
4. **MCP 서버 툴을 직접 부르는 것만 `async`**:
   `eqp_search_tool`, `location_search_tool`, `params_extract_tool`.

내가 처음에 추측했던 이름/시그니처와 실제가 다른 것들:

| 내 추측 | 실제 |
|---|---|
| `server_status_tool(server)` | `server_status_search_tool(fabs, target_dt, user_query)` |
| `sysadmin_tool(command)` | `sys_admin_tool(fabs, systems, user_query)` |
| `location_search_tool(carrier_id)` | `location_search_tool(identifier)` — 캐리어인지 랏인지 부르는 쪽이 확정하지 않는다 |
| `eqp_search_tool(eqp_id)` | `eqp_search_tool(machine_name, fab)` |
| `fab_extract_tool(text)` | `fab_extract_tool(user_query)` |
| 반환 `str` | 반환 `Dict[str, Any]` |
| `fab` 은 부수적 | **StatusAgent 툴 전부가 `fabs` 를 첫 인자로** 받는다 — 조회 범위를 공장 단위로 자르는 구조 |

### ⚠ `params_extract_tool` = 이미 존재하는 ID 판독기

`app/_tool.py 의 params_extract_tool` 이 "정체불명 ID 를 DB 로 판정" 하는데,
`params_extract_tool` 이 **이미 정확히 그 일을 한다** (캐리어/랏/장비/유닛/
포트/존 중 무엇인지, 아니면 unknown 인지를 DB 로 판정).

→ 그래서 app 쪽 판독기도 별도 모듈을 없애고 이 툴 하나로 합쳤다.
   사내 반입 시 이 툴의 **본문은 사내 것을 그대로 두고**, ActionAgent 가
`params_extract_tool` 을 부르게 바꾸는 게 맞습니다. 자세한 건
`docs/PORTING.md` 참고.

---

## 사용자가 직접 준 코드에 대해

타이핑 과정에서 난 오타(`StatesAgent`, `replasce`, `sace_supervisor_prompt`,
`clean_messagess`, `ㅏ우팅`, 닫히지 않은 괄호 등)는 **정정**했습니다.
구조와 로직은 손대지 않았습니다.

로직 중 원문 그대로 남긴 부분에는 `# [원본 그대로]` 주석을 달았습니다.
특히 `supervisor_node` 의 `final_message` 는 성공 경로에서 리스트,
실패 경로에서 문자열이 되는데, 이게 원문의 모양이라 고치지 않았습니다.
(`app/_node.py` 쪽에서는 정리되어 있습니다 — diff 로 보입니다.)

---

## 원본 그래프 (HITL 없음)

```
        START
          ↓
        Router ──── general ───→ GeneralAgent ──→ FinalGeneralAgent ──→ END
          │                          │
          │ supervisor               └── handoff ──┐
          ↓                                        │
        Supervisor ←─────────────────────────────┘
          │  ├─→ StatusAgent   ─┐
          │  ├─→ LocationAgent ─┤
          │  ├─→ LogAgent      ─┼─→ (실행 후 항상 Supervisor 복귀)
          │  ├─→ ExtractAgent  ─┤
          │  └─→ ActionAgent   ─┘
          ↓ FINISH
        FinalAnswerAgent ──→ END
```

원본의 `ActionAgent` 는 다른 워커와 **똑같은 모양**입니다 —
`create_react_agent` 하나에 `_util.agent_node` 위임. HITL 도, 파라미터 수집도,
승인 절차도 없습니다. 그 자리를 채우는 게 이번 작업입니다.

원본과 `app/` 의 실제 차이:

| 항목 | `origin/` | `app/` |
|---|---|---|
| ActionAgent | 일반 react agent | 턴 기반 HITL 단일 노드 |
| Supervisor | LLM 배분만 | needs-핸드오프 / Extract 선행 / HITL 턴 종료 우선순위 추가 |
| `AgentState` | messages/route/handoff/next/step/model_name | + `action`, `facts` |
| 체크포인터 | `MemorySaver` (빌더 안에서 생성) | 모듈 상수로 분리 — 모델별 그래프가 하나를 공유 |
| 모델 선택 | 상태에 `model_name` 은 있음 | + `/models` API · 모델별 그래프 캐시 |
| 스트림 | 최종 답변 토큰 | + `\x1e` 제어 프레임 (트레이스/HITL 질문) |
