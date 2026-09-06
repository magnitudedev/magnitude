"""Finite ACN RPC transport for a native Python host.

The generated contract owns payloads, results and recovery policies. This adapter
does not automatically replay requests, especially after a lost mutation response.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
import urllib.error
import urllib.request
from uuid import uuid4

ORIGIN = "http://127.0.0.1:10100"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class ClientError(Exception):
    pass


class ServiceUnavailable(ClientError):
    pass


class ProtocolMismatch(ClientError):
    pass


class UnknownOutcome(ClientError):
    pass


class RpcClient:
    def __init__(self, *, origin=ORIGIN, contract=None, starter=None):
        try:
            from jsonschema import Draft7Validator
        except ImportError as error:
            raise ClientError("This Hermes installation is missing JSON Schema support. Update Hermes with its standard installer, then retry.") from error
        self.origin = origin.rstrip("/")
        if contract is None:
            path = Path(__file__).parent / "dist" / "rpc-contract.json"
            try:
                contract = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise ClientError("Magnitude companion build is missing or invalid; reinstall it.") from error
        self.contract = contract
        self._health_validator = Draft7Validator(contract["health"])
        self._operations = {
            name: (operation, Draft7Validator(operation["payload"]),
                   Draft7Validator(operation["success"]), Draft7Validator(operation["error"]))
            for name, operation in contract["operations"].items()
        }
        self._starter = starter or self._start_service
        self._start_lock = Lock()
        # The local control connection must not inherit HTTP proxy settings.
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self):
        try:
            response = self._http.open(self.origin + "/health", timeout=2)
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise ClientError(f"Magnitude health returned HTTP {error.code}.") from error
            response = error
        except (urllib.error.URLError, OSError) as error:
            raise ServiceUnavailable("Magnitude service is unavailable.") from error
        try:
            with response:
                data = response.read(64 * 1024 + 1)
            if len(data) > 64 * 1024:
                raise ValueError("Oversized health response")
            health = json.loads(data)
            if not self._health_validator.is_valid(health):
                raise ValueError("Invalid health response")
        except (ValueError, OSError) as error:
            raise ClientError("Invalid Magnitude service health response.") from error
        if health.get("rpcVersion", 0) != self.contract["rpcVersion"]:
            raise ProtocolMismatch(
                "Magnitude companion and service protocols differ. Run "
                "`magnitude connections sync hermes`, then restart Hermes."
            )
        return health

    def _ready_health(self):
        health = self.health()
        if health["state"]["_tag"] != "Ready":
            raise ServiceUnavailable("Magnitude service is not ready.")
        return health

    def _connect(self):
        try:
            return self._ready_health()
        except ServiceUnavailable:
            with self._start_lock:
                try:
                    return self._ready_health()
                except ServiceUnavailable:
                    self._starter()
                    return self._ready_health()

    @staticmethod
    def _start_service():
        executable = os.environ.get("MAGNITUDE_CLI", "").strip() or "magnitude"
        with tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run(
                    [executable, "service", "start"], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=stderr, timeout=600, check=False,
                )
            except FileNotFoundError as error:
                raise ClientError("Magnitude CLI is not installed. Run /magnitude-setup.") from error
            except subprocess.TimeoutExpired as error:
                raise ClientError("Magnitude service startup timed out. Run `magnitude service status`.") from error
            except OSError as error:
                raise ClientError("Could not launch Magnitude. Check that its CLI is executable.") from error
            if result.returncode:
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 4096))
                detail = stderr.read().decode("utf-8", errors="replace").strip()
                raise ClientError(f"Magnitude service startup failed: {detail or result.returncode}")

    def call(self, name, payload, *, start_service=True, timeout=600):
        operation, request_validator, response_validator, error_validator = self._operations[name]
        if not request_validator.is_valid(payload):
            raise ClientError("Invalid Magnitude command arguments. Use the exact model ID from /magnitude.")
        health = self._connect() if start_service else self._ready_health()
        # Effect RPC decodes wire request IDs as BigInt, not arbitrary strings.
        request_id = str(uuid4().int)
        envelope = {
            "_tag": "Request", "id": request_id, "tag": operation["tag"],
            "payload": payload, "headers": [],
        }
        request = urllib.request.Request(
            self.origin + "/rpc", data=(json.dumps(envelope) + "\n").encode(),
            headers={"Content-Type": "application/ndjson", "x-magnitude-acn-id": health["id"]},
        )
        try:
            with self._http.open(request, timeout=timeout) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Oversized RPC response")
            frames = [json.loads(line) for line in body.splitlines() if line.strip()]
            if len(frames) != 1:
                raise ValueError("Expected one finite RPC exit")
            frame = frames[0]
            if frame.get("_tag") != "Exit" or frame.get("requestId") != request_id:
                raise ValueError("Mismatched RPC exit")
            outcome = frame["exit"]
            if outcome.get("_tag") == "Success":
                if not response_validator.is_valid(outcome.get("value")):
                    raise ValueError("Invalid RPC result")
                return outcome["value"]
            if outcome.get("_tag") != "Failure":
                raise ValueError("Invalid RPC outcome")
            cause = outcome.get("cause")
            # Only a validated domain failure is a negative acknowledgement.
            # Defects, interruptions and malformed failures leave a mutation's
            # outcome uncertain, just like a disconnected response does.
            if not isinstance(cause, dict) or cause.get("_tag") != "Fail" \
                    or not error_validator.is_valid(cause.get("error")):
                raise ValueError("RPC did not acknowledge a domain outcome")
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            if operation["recovery"] == "AtMostOnce":
                raise UnknownOutcome(
                    "Magnitude did not return a valid acknowledgement. The operation may have taken effect; "
                    "check /magnitude before retrying. It was not replayed."
                ) from error
            raise ClientError("Magnitude request failed. Check the service and retry.") from error
        raise ClientError("Magnitude rejected the operation: " + cause["error"]["message"])
