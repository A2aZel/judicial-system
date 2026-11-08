from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class InvalidContentLength(Exception):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            content_length = self._content_length(scope)
        except InvalidContentLength:
            await self._response(scope, receive, send, 400, "Invalid Content-Length")
            return
        if content_length is not None and content_length > self.max_bytes:
            await self._response(scope, receive, send, 413, "Request body too large")
            return

        received_bytes = 0
        too_large = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        async def controlled_send(message: Message) -> None:
            nonlocal response_started
            if too_large:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, controlled_send)
        if too_large:
            if response_started:
                raise RuntimeError("request body limit exceeded after response start")
            await self._response(scope, receive, send, 413, "Request body too large")

    @staticmethod
    async def _response(
        scope: Scope,
        receive: Receive,
        send: Send,
        status: int,
        detail: str,
    ) -> None:
        response = JSONResponse(status_code=status, content={"detail": detail})
        await response(scope, receive, send)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_value = headers.get(b"content-length")
        if raw_value is None:
            return None
        try:
            value = int(raw_value)
            if value < 0:
                raise ValueError
            return value
        except ValueError as exc:
            raise InvalidContentLength from exc
