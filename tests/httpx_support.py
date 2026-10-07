"""Small HTTPX-boundary response double for deterministic timeline tests."""
import httpx


class ResponseMixin:
    http_version = "HTTP/1.1"
    status_code = 200
    extensions = {}

    @property
    def is_error(self):
        return self.status_code >= 400

    @property
    def content(self):
        return self.payload

    def raise_for_status(self):
        if self.is_error:
            response = httpx.Response(self.status_code, content=self.payload,
                                      request=httpx.Request("GET", "https://example.test"))
            response.raise_for_status()

    @property
    def stream(self):
        return self.iter_bytes()

    def iter_bytes(self):
        for line in self:
            yield line

    def close(self):
        pass
