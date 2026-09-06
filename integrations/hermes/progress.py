"""Request-scoped presentation driven by Hermes's public observer hooks."""

from dataclasses import dataclass
import logging
from threading import Event, Lock, Thread
import time
from urllib.parse import urlsplit
from uuid import uuid4

from .terminal import TerminalOutput, TerminalStatus, plain_line
from .observations import ObservationReader, session_group, phase_text, timing_text

logger = logging.getLogger(__name__)


def is_magnitude_endpoint(base_url):
    if not isinstance(base_url, str):
        return False
    try:
        url = urlsplit(base_url)
        return (
            url.scheme == "http" and url.hostname == "127.0.0.1" and url.port == 10100
            and url.path.rstrip("/") == "/inference/v1"
            and url.username is None and url.password is None
            and not url.query and not url.fragment
        )
    except ValueError:
        return False


@dataclass(frozen=True)
class RequestPresentation:
    terminal: TerminalStatus
    label: str
    started: float
    group_id: str | None = None
    observation_id: str | None = None


class InferenceProgress:
    def __init__(self, capture=TerminalOutput.capture, clock=time.monotonic, reader=None):
        self._capture = capture
        self._clock = clock
        self._requests = {}
        self._lock = Lock()
        self._closed = False
        self._reader = reader or ObservationReader()
        self._stop = Event()
        self._worker = None

    @staticmethod
    def _key(event):
        session = event.get("session_id")
        request = event.get("api_request_id")
        if isinstance(session, str) and session and isinstance(request, str) and request:
            return session, request
        return None

    def start(self, **event):
        if not is_magnitude_endpoint(event.get("base_url")):
            return
        key = self._key(event)
        output = self._capture()
        if key is None or output is None:
            return
        # Labels make interleaved background sessions explicit rather than attributing
        # their work to the foreground conversation. No prompt text enters the label.
        label = plain_line(f"{event.get('model', 'Local model')} · request {key[1][:8]}")
        presentation = RequestPresentation(TerminalStatus(output), label, self._clock(),
                                           event.get("group_id"), event.get("observation_id"))
        with self._lock:
            if self._closed or key in self._requests or len(self._requests) >= 128:
                return
            self._requests[key] = presentation
            if presentation.group_id and self._worker is None:
                self._worker = Thread(target=self._poll, name="magnitude-progress", daemon=True)
                self._worker.start()
        presentation.terminal.phase("requesting", f"{label} · Requesting")

    def finish(self, *, failed=False, **event):
        key = self._key(event)
        with self._lock:
            presentation = self._requests.get(key)
        if presentation is not None:
            elapsed = max(0, self._clock() - presentation.started)
            result = "Request failed" if failed else "Request finished"
            summary = ""
            try:
                if not failed and presentation.group_id:
                    observations = self._reader.read(presentation.group_id) or []
                    observation = next((item for item in observations if item["requestId"] == presentation.observation_id), None)
                    if observation and observation.get("timings"):
                        summary = " · " + timing_text(observation["timings"])
            finally:
                with self._lock:
                    current = self._requests.get(key)
                    if current is presentation:
                        self._requests.pop(key)
                    presentation.terminal.finish(
                        f"{presentation.label} · {result} · {elapsed:.1f}s{summary}"
                        if current is presentation and not self._closed else None)

    def _poll(self):
        while not self._stop.wait(0.25):
            with self._lock:
                presentations = list(self._requests.values())
            groups = {item.group_id for item in presentations if item.group_id}
            for group in groups:
                if self._stop.is_set():
                    return
                try:
                    observations = self._reader.read(group) or []
                    for presentation in presentations:
                        if presentation.group_id != group:
                            continue
                        observation = next((item for item in observations if item["requestId"] == presentation.observation_id), None)
                        if observation and observation.get("progress"):
                            progress = observation["progress"]
                            presentation.terminal.phase(progress["phase"], f"{presentation.label} · {phase_text(progress)}")
                except Exception:
                    logger.debug("Magnitude progress observation failed", exc_info=True)

    def end_session(self, **event):
        with self._lock:
            keys = [key for key in self._requests if key[0] == event.get("session_id")]
            presentations = [self._requests.pop(key) for key in keys]
        for presentation in presentations:
            presentation.terminal.finish()

    def close(self):
        with self._lock:
            self._closed = True
            presentations = list(self._requests.values())
            self._requests.clear()
            worker = self._worker
        self._stop.set()
        for presentation in presentations:
            presentation.terminal.finish()
        if worker is not None:
            worker.join(timeout=3)

    def execute(self, *, request, next_call, **event):
        # The public execution middleware gives each actual attempt a fresh ID.
        # The stock transport/parser still executes exactly once and owns errors.
        if not is_magnitude_endpoint(event.get("base_url")) or self._key(event) is None:
            return next_call(request)
        observation_id = str(uuid4())
        group_id = session_group(event["session_id"])
        headers = {key: value for key, value in (request.get("extra_headers") or {}).items()
                   if key.lower() not in ("magnitude-include-progress", "magnitude-observation-id", "magnitude-observation-group")}
        headers.update({"Magnitude-Include-Progress": "true", "Magnitude-Observation-Id": observation_id,
                        "Magnitude-Observation-Group": group_id})
        correlated = dict(request, extra_headers=headers)
        presentation_event = dict(event, api_request_id=observation_id, observation_id=observation_id, group_id=group_id)
        try:
            self.start(**presentation_event)
        except Exception:
            logger.debug("Magnitude progress presentation failed", exc_info=True)
        failed = True
        try:
            result = next_call(correlated)
            failed = False
            return result
        finally:
            try:
                self.finish(failed=failed, **presentation_event)
            except Exception:
                logger.debug("Magnitude progress completion failed", exc_info=True)


def register_progress(ctx):
    progress = InferenceProgress()

    def observer(callback):
        def invoke(**event):
            try:
                callback(**event)
            except Exception:
                # Presentation is an optional observer, never a reason to fail inference.
                logger.debug("Magnitude progress presentation failed", exc_info=True)
        return invoke

    ctx.register_middleware("llm_execution", progress.execute)
    ctx.register_hook("on_session_end", observer(progress.end_session))
    ctx.register_hook("on_session_reset", observer(progress.end_session))
    ctx.on_unload(progress.close)
