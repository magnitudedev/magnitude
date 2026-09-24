#!/usr/bin/env python3
"""End-to-end checks of prefix retention and elastic state against a running V4 engine.

Start the engine first (for `repeat --image`, with `--projector`), then run one mode:

  sharing PORT         shared system prompt: a second question plans a branch, later
                       questions, follow-up turns and four concurrent questions resume
                       from it; four simultaneous requests on an unseen prefix compute it
                       once (three attach: cached > 0).
  repeat PORT [IMAGE]  a request whose prompt equals a retained prompt exactly: the same
                       request twice (identical greedy output, cached < prompt) and the
                       same prompt with other sampling, for text and optionally an image.
  timing PORT [TOKENS] decode step host time of one streamed request: odd and even step
                       medians within 25%, every 50-step window within 1.5x of the first.
                       Run it on a freshly started engine with no other GPU work running.

Each mode prints its measurements and exits nonzero on failure. Greedy sampling throughout.
"""
import base64
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

MODEL = "Qwen3.5-4B-Q4_K_M"


def post(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=600)


def chat(port, messages, max_tokens):
    started = time.time()
    try:
        with post(port, {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        return {"error": error.code, "body": error.read().decode()[:300]}
    return {
        "prompt": data["usage"]["prompt_tokens"],
        "cached": data["usage"]["prompt_tokens_details"]["cached_tokens"],
        "seconds": round(time.time() - started, 2),
        "text": data["choices"][0]["message"]["content"].strip(),
    }


def report(results):
    for name, result in results.items():
        shown = dict(result, text=result.get("text", "")[:100].replace("\n", " "))
        print(name, json.dumps(shown))


def sharing(port):
    system = "You are a meticulous assistant for a hardware company. " + " ".join(
        f"Policy {i}: answer precisely, cite the relevant product line, avoid speculation, "
        "and keep answers under three sentences unless asked otherwise."
        for i in range(40)
    )

    def ask(prefix, question):
        return chat(port, [{"role": "system", "content": prefix}, {"role": "user", "content": question}], 48)

    results = {
        "first": ask(system, "Why is the sky blue?"),
        "second": ask(system, "What is the boiling point of water at sea level?"),
        "third": ask(system, "Name three primary colors."),
    }
    turn = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": results["first"].get("text", "")},
    ]
    results["follow_a"] = chat(port, turn + [{"role": "user", "content": "Explain it to a child."}], 48)
    results["follow_b"] = chat(port, turn + [{"role": "user", "content": "Is it blue on Mars?"}], 48)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        warm = list(pool.map(lambda n: ask(system, f"Give one fact about the number {n}."), range(4)))
    cold_prefix = system.replace("hardware company", f"shipping company ({time.time()})")
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        cold = list(pool.map(lambda n: ask(cold_prefix, f"Name a port on continent {n}."), range(4)))
    results.update({f"concurrent_{i}": r for i, r in enumerate(warm)})
    results.update({f"cold_{i}": r for i, r in enumerate(cold)})
    report(results)
    failures = [name for name, r in results.items() if "error" in r or not r["text"]]
    if any(results[name].get("cached", 0) == 0 for name in ("third", "follow_a", "follow_b")):
        failures.append("later requests did not resume from the shared prefix")
    if sum(1 for r in cold if r.get("cached", 0) > 0) < 3:
        failures.append("simultaneous arrivals computed the unseen prefix more than once")
    return failures


def repeat(port, image=None):
    def case(name, messages):
        first, second, other = chat(port, messages, 24), chat(port, messages, 24), chat(port, messages, 12)
        report({f"{name}_first": first, f"{name}_second": second, f"{name}_other": other})
        failures = [f"{name}_{label}" for label, r in (("first", first), ("second", second), ("other", other)) if "error" in r]
        if failures:
            return failures
        if first["prompt"] < 64:
            failures.append(f"{name}: prompt {first['prompt']} is below the retention minimum")
        if second["text"] != first["text"]:
            failures.append(f"{name}: the repeat changed the greedy output")
        if not 0 < second["cached"] < second["prompt"]:
            failures.append(f"{name}: the repeat resumed {second['cached']} of {second['prompt']} tokens")
        if not other["text"]:
            failures.append(f"{name}: the same prompt with other sampling produced nothing")
        return failures

    text = " ".join(f"Item {i} of the inventory is a numbered brass fitting." for i in range(20))
    failures = case("text", [{"role": "user", "content": text + " Summarize the inventory in one sentence."}])
    if image:
        with open(image, "rb") as stream:
            data = base64.b64encode(stream.read()).decode()
        failures += case(
            "image",
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                {"type": "text", "text": "Describe this image in one sentence."},
            ]}],
        )
    return failures


def timing(port, tokens=300):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a long, detailed story about a lighthouse keeper."}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": True,
    }
    arrivals = []
    with post(port, body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            choices = json.loads(line[5:]).get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                arrivals.append(time.perf_counter())
    intervals = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    if len(intervals) < 20:
        return [f"only {len(arrivals)} streamed chunks"]
    windows = [statistics.median(w) for w in (intervals[i : i + 50] for i in range(0, len(intervals), 50)) if len(w) >= 10]
    odd, even = statistics.median(intervals[1::2]), statistics.median(intervals[0::2])
    print(f"chunks={len(arrivals)} median_ms={statistics.median(intervals):.2f} max_ms={max(intervals):.2f}")
    print("window_medians_ms=" + " ".join(f"{m:.2f}" for m in windows))
    print(f"odd_median_ms={odd:.2f} even_median_ms={even:.2f}")
    failures = []
    if max(odd, even) > 1.25 * min(odd, even):
        failures.append("odd and even decode steps differ (alternating host work)")
    if max(windows) > 1.5 * windows[0]:
        failures.append("decode step time grows across the request")
    return failures


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("sharing", "repeat", "timing"):
        raise SystemExit(__doc__)
    mode, port, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    if mode == "sharing":
        failures = sharing(port)
    elif mode == "repeat":
        failures = repeat(port, extra[0] if extra else None)
    else:
        failures = timing(port, int(extra[0]) if extra else 300)
    for failure in failures:
        print("FAIL", failure)
    print("PASS" if not failures else "FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
