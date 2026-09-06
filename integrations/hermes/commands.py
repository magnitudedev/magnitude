"""Hermes owns command execution and output; the companion owns model controls."""

from .terminal import plain_line
from threading import Lock


class ModelCommands:
    def __init__(self, client_factory=None):
        self._client = None
        self._client_factory = client_factory
        self._client_lock = Lock()

    def _call(self, operation, payload, present):
        from .client import ClientError, RpcClient

        try:
            with self._client_lock:
                if self._client is None:
                    self._client = (self._client_factory or RpcClient)()
                client = self._client
            return present(client.call(operation, payload))
        except ClientError as error:
            return "Magnitude: " + plain_line(str(error), 4096)

    def status(self, args):
        if args.strip():
            return "Usage: /magnitude"
        return self._call("status", {}, format_catalog)

    def load(self, args):
        model_id = args.strip()
        if not model_id:
            return "Usage: /load-model <model-id> — list installed models with /magnitude."
        return self._call("load", {"modelId": model_id}, lambda _: "Loaded " + plain_line(model_id))

    def stop(self, args):
        if args.strip():
            return "Usage: /stop-model"
        return self._call("stop", {}, lambda _: "Stopped the active Magnitude model.")


def format_catalog(catalog):
    if catalog["_tag"] == "Initializing":
        return "Magnitude is discovering models. Try /magnitude again shortly."
    lines = []
    for entry in catalog["models"]:
        if entry["_tag"] != "Local":
            continue
        model = entry["product"]
        state = model["state"] if model["_tag"] == "Discovered" else model["acquisitionState"]
        phase = state["_tag"]
        if phase == "NotInstalled":
            continue
        if phase in ("Ready", "Installed", "UpdateAvailable"):
            phase = state["residencyState"]["_tag"]
        label = {
            "Installing": "Downloading", "InstallFailed": "Download failed", "Updating": "Updating",
            "UpdateFailed": "Update failed", "Removing": "Removing", "RemoveFailed": "Removal failed",
            "UpdateAvailable": "Update available", "Unavailable": "Unavailable", "Unloaded": "Unloaded",
            "Requested": "Load requested", "Loading": "Loading", "Ready": "Ready", "Stopping": "Stopping",
            "Failed": "Load failed",
        }[phase]
        if state["_tag"] == "UpdateAvailable":
            label += " · Update available"
        lines.append(plain_line(f"{model['modelId']} · {label}", 1024))
    return "Magnitude local models:\n" + "\n".join(sorted(lines)) if lines else "No local models found. Run /magnitude-setup."
