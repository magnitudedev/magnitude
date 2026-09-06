"""Test-only disposable HTTP process; this is never an inference engine adapter."""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ready"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["max_tokens"] == (32768 if body.get("tools") else 256)
        events = [
            {
                "id": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "echo-0",
                                    "function": {"name": "echo", "arguments": '{"value":7}'},
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "id": "fixture",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "fixture",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "timings": {
                    "prompt_n": 10,
                    "cache_n": 0,
                    "predicted_n": 4,
                    "prompt_ms": 20,
                    "predicted_ms": 40,
                },
            },
        ]
        if not body.get("tools"):
            events[0]["choices"][0]["delta"] = {"content": "The story continues."}
            events[1]["choices"][0]["finish_reason"] = "stop"
        stream = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        stream += "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(stream.encode())))
        self.end_headers()
        self.wfile.write(stream.encode())


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
