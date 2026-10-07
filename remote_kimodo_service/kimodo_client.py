"""HTTP client for the Spark's Kimodo service. stdlib only (the mirroring venv has no
requests/httpx, and this needs nothing they add).

Every failure leaves here as one of two exceptions with a one-line message, never a
traceback: SparkUnavailable (try again later / play from the cache) or GenerationFailed
(the service answered, and said no). Callers never block the player on this: it runs
on the generation worker thread only.

Over the SSH tunnel, "service down" does NOT look like "connection refused": ssh
accepts the local connection, fails to reach 127.0.0.1:8765 on the Spark, and closes
it. That arrives as RemoteDisconnected / ConnectionResetError. Refused means the
tunnel itself is not running. On Windows, refused on localhost takes 2.05 s (the OS
retries the SYN; measured), so a timeout shorter than that reports a timeout instead.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .kimodo_contract import GenerationRequest

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_HEALTH_TIMEOUT_S = 3.0
START_HINT = "start the service on the Spark and the SSH tunnel, see remote_kimodo_service/README.md"


class SparkUnavailable(Exception):
    """Nothing answered (tunnel down, service down, timeout). Not the user's fault."""


class GenerationFailed(Exception):
    """The service answered with an error (bad request, generation crashed)."""


@dataclass
class GenerationResult:
    data: bytes
    total_s: float              # measured here: request sent -> last byte received
    generation_s: float | None  # measured on the Spark
    queue_s: float | None       # time the request waited for the service's lock
    server_cache: bool

    @property
    def transfer_s(self) -> float | None:
        """What the link cost: total minus the Spark's own time."""
        if self.generation_s is None:
            return None
        return max(0.0, self.total_s - self.generation_s - (self.queue_s or 0.0))


def _float_header(headers, name: str) -> float | None:
    try:
        return float(headers.get(name))
    except (TypeError, ValueError):
        return None


class KimodoClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT_S,
                 health_timeout: float = DEFAULT_HEALTH_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.health_timeout = health_timeout

    def _open(self, req: urllib.request.Request, timeout: float):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise GenerationFailed(f"service answered {e.code}: {self._detail(e)}") from None
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, ConnectionRefusedError):
                raise SparkUnavailable(f"connection refused at {self.base_url} - is the SSH tunnel running? "
                                       f"({START_HINT})") from None
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise SparkUnavailable(f"no answer from {self.base_url} within {timeout:g} s") from None
            raise SparkUnavailable(f"cannot reach {self.base_url}: {reason} ({START_HINT})") from None
        except (http.client.RemoteDisconnected, ConnectionResetError, ConnectionAbortedError):
            raise SparkUnavailable(f"{self.base_url} accepted and closed the connection - tunnel up, "
                                   f"service not running? ({START_HINT})") from None
        except (TimeoutError, socket.timeout):
            raise SparkUnavailable(f"no answer from {self.base_url} within {timeout:g} s") from None
        except (OSError, http.client.HTTPException) as e:
            raise SparkUnavailable(f"network error talking to {self.base_url}: {type(e).__name__}: {e}") from None

    @staticmethod
    def _detail(e: urllib.error.HTTPError) -> str:
        try:
            body = e.read().decode("utf-8", "replace")
            return str(json.loads(body).get("detail", body))[:300]
        except Exception:
            return e.reason or "no detail"

    def _read(self, resp, timeout: float) -> bytes:
        try:
            with resp:
                return resp.read()
        except (TimeoutError, socket.timeout):
            raise SparkUnavailable(f"transfer from {self.base_url} stalled for {timeout:g} s") from None
        except (http.client.IncompleteRead, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            raise SparkUnavailable(f"transfer from {self.base_url} broke off: {type(e).__name__}") from None

    def health(self) -> dict:
        req = urllib.request.Request(f"{self.base_url}/health", method="GET")
        body = self._read(self._open(req, self.health_timeout), self.health_timeout)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise GenerationFailed(f"/health did not return JSON ({len(body)} bytes)") from None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = json.dumps(request.to_json()).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/generate", data=payload, method="POST",
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        resp = self._open(req, self.timeout)
        headers = resp.headers
        data = self._read(resp, self.timeout)
        return GenerationResult(data=data, total_s=time.perf_counter() - t0,
                                generation_s=_float_header(headers, "X-Generation-Seconds"),
                                queue_s=_float_header(headers, "X-Queue-Seconds"),
                                server_cache=headers.get("X-Cache") == "hit")
