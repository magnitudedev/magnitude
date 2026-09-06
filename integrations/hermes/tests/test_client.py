import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest


spec = importlib.util.spec_from_file_location("magnitude_client", Path(__file__).parents[1] / "client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.started = 0
        self.mode = "success"
        self.health = {
            "service": "magnitude-acn", "version": "test", "revision": 1,
            "id": "test-instance", "pid": 1, "state": {"_tag": "Ready"},
        }
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply(200, json.dumps(test.health).encode())

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not isinstance(request.get("id"), str) or not request["id"].isdecimal():
                    self.reply(200, b'{"_tag":"Defect","defect":{"message":"Failed to parse String to BigInt"}}\n')
                    return
                test.calls.append((request, self.headers.get("x-magnitude-acn-id")))
                value = {"_tag": "Initializing"} if request["tag"] == "GetModelCatalog" else {}
                frame = {"_tag": "Exit", "requestId": request["id"], "exit": {"_tag": "Success", "value": value}}
                if test.mode == "wrong-id":
                    frame["requestId"] = "another-request"
                if test.mode == "invalid-value":
                    frame["exit"]["value"] = {"_tag": "NotACatalog"}
                if test.mode == "failure":
                    frame["exit"] = {"_tag": "Failure", "cause": {"_tag": "Fail", "error": {
                        "_tag": "LocalModelMutationFailed", "code": "fixture", "message": "Fixture refusal", "retryable": False,
                    }}}
                if test.mode == "malformed-failure":
                    frame["exit"] = {"_tag": "Failure", "cause": []}
                if test.mode == "defect":
                    frame["exit"] = {"_tag": "Failure", "cause": {"_tag": "Die", "defect": "fixture"}}
                body = json.dumps(frame).encode() + b"\n"
                if test.mode == "malformed":
                    body = b"not-json\n"
                if test.mode == "duplicate":
                    body += body
                if test.mode == "empty":
                    body = b""
                self.reply(200, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.rpc = client.RpcClient(origin=f"http://127.0.0.1:{self.server.server_port}", starter=self.start)
        self.health["rpcVersion"] = self.rpc.contract["rpcVersion"]

    def start(self):
        self.started += 1

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_query_is_fenced_and_generated_tag_is_used(self):
        self.assertEqual(self.rpc.call("status", {}), {"_tag": "Initializing"})
        self.assertEqual(self.calls[0][0]["tag"], "GetModelCatalog")
        self.assertEqual(self.calls[0][1], "test-instance")
        self.assertTrue(self.calls[0][0]["id"].isdecimal())
        self.assertEqual(self.started, 0)

    def test_loading_uses_model_id(self):
        self.rpc.call("load", {"modelId": "hf:fixture/model/model.gguf"})
        self.assertEqual(self.calls[0][0]["payload"], {"modelId": "hf:fixture/model/model.gguf"})

    def test_invalid_payload_is_rejected_before_io(self):
        with self.assertRaises(client.ClientError):
            self.rpc.call("load", {"modelId": 123})
        self.assertEqual(self.calls, [])

    def test_protocol_mismatch_does_not_start_or_mutate(self):
        self.health["rpcVersion"] += 1
        with self.assertRaises(client.ProtocolMismatch):
            self.rpc.call("stop", {})
        self.assertEqual(self.calls, [])
        self.assertEqual(self.started, 0)

    def test_missing_protocol_version_is_incompatible(self):
        del self.health["rpcVersion"]
        with self.assertRaises(client.ProtocolMismatch):
            self.rpc.health()

    def test_invalid_service_is_not_bootstrapped(self):
        self.health["service"] = "unrelated-service"
        with self.assertRaises(client.ClientError):
            self.rpc.call("stop", {})
        self.assertEqual(self.started, 0)

    def test_uncertain_mutations_are_never_replayed(self):
        for mode in ("wrong-id", "malformed", "duplicate", "empty", "malformed-failure", "defect"):
            with self.subTest(mode=mode):
                self.mode = mode
                before = len(self.calls)
                with self.assertRaises(client.UnknownOutcome):
                    self.rpc.call("stop", {})
                self.assertEqual(len(self.calls), before + 1)

    def test_query_result_validation(self):
        self.mode = "invalid-value"
        with self.assertRaises(client.ClientError):
            self.rpc.call("status", {})

    def test_received_failure_is_not_reported_as_unknown_outcome(self):
        self.mode = "failure"
        with self.assertRaises(client.ClientError) as raised:
            self.rpc.call("stop", {})
        self.assertNotIsInstance(raised.exception, client.UnknownOutcome)

    def test_explicit_command_waits_for_service_lifecycle_before_rpc(self):
        self.health["state"] = {"_tag": "Starting", "activity": "Resolving"}
        def ready():
            self.started += 1
            self.health["state"] = {"_tag": "Ready"}
        self.rpc._starter = ready
        self.assertEqual(self.rpc.call("status", {}), {"_tag": "Initializing"})
        self.assertEqual(self.started, 1)
        self.assertEqual(len(self.calls), 1)

    def test_passive_read_never_bootstraps_or_sends_rpc_to_starting_service(self):
        self.health["state"] = {"_tag": "Starting", "activity": "Resolving"}
        with self.assertRaises(client.ServiceUnavailable):
            self.rpc.call("status", {}, start_service=False)
        self.assertEqual(self.started, 0)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
