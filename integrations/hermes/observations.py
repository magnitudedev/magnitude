"""Read-only progress access. Observing never starts or repairs a service."""

from threading import Lock
from uuid import NAMESPACE_URL, uuid5

from .client import ClientError, RpcClient


def session_group(session_id):
    from hermes_constants import get_hermes_home
    # Separate identically named sessions in different Hermes profiles/homes.
    return str(uuid5(NAMESPACE_URL, f"magnitude:hermes:{get_hermes_home().resolve()}:{session_id}"))


class ObservationReader:
    def __init__(self):
        self._client = None
        self._lock = Lock()

    def read(self, group_id):
        try:
            with self._lock:
                if self._client is None:
                    self._client = RpcClient()
                client = self._client
            return client.call("observations", {"groupId": group_id}, start_service=False, timeout=2)
        except ClientError:
            return None


def phase_text(progress):
    phase = progress["phase"]
    if phase == "model_loading":
        return f"Loading model · {max(0, min(100, progress['fraction'] * 100)):.0f}%"
    if phase == "prefill":
        return f"Prefill · {progress['completed_tokens']}/{progress['total_tokens']} tokens · {progress['cached_tokens']} cached"
    return {"queued": "Queued", "preparing": "Preparing", "generating": "Generating"}[phase]


def timing_text(timings):
    return (f"prefill {timings['prompt_ms'] / 1000:.2f}s · "
            f"first token {timings['time_to_first_token_ms'] / 1000:.2f}s · "
            f"{timings['predicted_n']} tokens · {timings['predicted_per_second']:.1f} tok/s")
