"""MCP 커넥션 매니저 (스텁).

사내에서는 여기서 실제 MCP 서버들에 붙는다.
이 저장소는 사내망 밖이라 실제 연결은 하지 않고, 서버 기동/종료 훅의
자리와 호출 규약만 맞춰 둔다.

main.py 의 lifespan 이 이걸 쓴다.
    startup  -> await mcp_manager.connect()
    shutdown -> await mcp_manager.disconnect()

사내 반입 시 connect/disconnect 본문만 실제 구현으로 갈아끼우면 된다.
"""


class MCPManager:
    """MCP 서버 연결을 한 곳에서 들고 있는 매니저."""

    def __init__(self):
        # 연결된 세션들 (사내에서는 서버명 -> 세션 객체)
        self.sessions: dict = {}

        # 연결 여부. 중복 connect/disconnect 를 막는 용도
        self.connected: bool = False

    async def connect(self):
        """서버 기동 시 1회 호출. MCP 서버들에 붙는다."""
        if self.connected:
            print("[MCP] 이미 연결되어 있어 건너뜁니다.", flush=True)
            return

        # 사내 구현 자리:
        #   for name, cfg in MCP_SERVERS.items():
        #       self.sessions[name] = await open_session(cfg)
        print("[MCP] connect (스텁 — 사내에서 실제 서버 연결로 교체)", flush=True)
        self.connected = True

    async def disconnect(self):
        """서버 종료 시 1회 호출. 열린 세션을 정리한다."""
        if not self.connected:
            return

        # 사내 구현 자리:
        #   for session in self.sessions.values():
        #       await session.close()
        print("[MCP] disconnect (스텁)", flush=True)
        self.sessions.clear()
        self.connected = False


# 앱 전체가 공유하는 싱글턴
mcp_manager = MCPManager()
