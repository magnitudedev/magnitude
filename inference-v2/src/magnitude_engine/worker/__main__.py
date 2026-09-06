"""Private development worker entry point; never opens a network listener."""

import argparse
import os
import queue
import resource
import sys
from contextlib import ExitStack
from dataclasses import asdict
from threading import Event, Thread

from magnitude_engine.composition import build, digest, dumps, loads
from magnitude_engine.engine.blueprint import Engine as EngineBlueprint

from .framing import Frame, read_frame, write_frame


def parent_watchdog(parent: int, stop: Event) -> None:
    while not stop.wait(0.1):
        if os.getppid() != parent:
            os._exit(70)


def run(parent: int) -> int:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    stop, wake, retired = Event(), Event(), Event()
    Thread(target=parent_watchdog, args=(parent, retired), daemon=True).start()
    first = read_frame(sys.stdin.buffer)
    if set(first.message) != {"type", "blueprint", "digest"} or first.message["type"] != "load":
        raise ValueError("first worker command must describe one engine residency")
    blueprint = loads(first.message["blueprint"])
    if not isinstance(blueprint, EngineBlueprint) or digest(blueprint) != first.message["digest"]:
        raise ValueError("worker load requires a verified engine blueprint")
    incoming: queue.Queue[dict] = queue.Queue(64)
    outgoing: queue.Queue[Frame | None] = queue.Queue(64)
    failures = []

    def receive() -> None:
        try:
            while not stop.is_set():
                message = read_frame(sys.stdin.buffer, first.generation).message
                if message == {"type": "shutdown"}:
                    stop.set()
                    break
                while not stop.is_set():
                    try:
                        incoming.put(message, timeout=0.1)
                        wake.set()
                        break
                    except queue.Full:
                        continue
        except EOFError:
            stop.set()
        except BaseException as error:
            failures.append(error)
            stop.set()
        finally:
            wake.set()

    def transmit() -> None:
        try:
            while True:
                frame = outgoing.get()
                if frame is None:
                    return
                write_frame(sys.stdout.buffer, frame)
        except BaseException as error:
            failures.append(error)
            stop.set()
            wake.set()

    def send(message: dict) -> None:
        outgoing.put_nowait(Frame(first.generation, message))

    Thread(target=receive, daemon=True).start()
    writer = Thread(target=transmit, daemon=True)
    writer.start()
    runtime = None
    lifetime = ExitStack()
    try:
        # MLX is imported and initialized only in this disposable process.
        from magnitude_engine.engine.delivery import Finished, PrefillProgress
        from magnitude_engine.engine.requests import GenerationRequest
        from magnitude_engine.generation.constraint_spec import ConstraintSpec
        from magnitude_engine.generation.sampling_policy import SamplingPolicy

        runtime = lifetime.enter_context(build(blueprint))
        send(
            {
                "type": "ready",
                "pid": os.getpid(),
                **runtime.properties,
                "composition_digest": digest(blueprint),
                "composition": blueprint.describe(),
                "composition_json": dumps(blueprint),
            }
        )
        handles = {}
        reading: set[str] = set()
        cancelling: set[str] = set()
        while not stop.is_set():
            wake.clear()
            # Bound control work per iteration so input cannot starve generation.
            for _ in range(16):
                if outgoing.full():
                    break
                try:
                    message = incoming.get_nowait()
                except queue.Empty:
                    break
                kind, identity = message.get("type"), message.get("request_id")
                if not isinstance(identity, str) or not 1 <= len(identity) <= 128:
                    raise ValueError("worker request identity is invalid")
                if kind == "infer":
                    if identity in handles:
                        raise ValueError("duplicate live worker request")
                    try:
                        if set(message) != {
                            "type",
                            "request_id",
                            "prompt",
                            "sampling",
                            "max_tokens",
                            "stop_tokens",
                            "constraint",
                            "progress",
                        }:
                            raise ValueError("inference command fields differ from protocol")
                        if type(message["progress"]) is not bool:
                            raise ValueError("progress subscription must be boolean")
                        request = GenerationRequest(
                            tuple(message["prompt"]),
                            SamplingPolicy(**message["sampling"]),
                            message["max_tokens"],
                            tuple(message["stop_tokens"]),
                            None
                            if message["constraint"] is None
                            else ConstraintSpec(**message["constraint"]),
                        )
                        if any(
                            token >= runtime.properties["vocab_size"] for token in request.prompt
                        ):
                            raise ValueError("input token is outside the target vocabulary")
                        if (
                            len(request.prompt) + request.max_tokens
                            > runtime.properties["context_tokens"]
                        ):
                            raise ValueError("request exceeds the configured context capacity")
                        if len(handles) >= 1024:
                            raise OverflowError("worker request delivery capacity is full")
                        handles[identity] = runtime.engine.submit(
                            request,
                            identity=identity,
                            output_capacity=runtime.output_capacity,
                            progress=message["progress"],
                        )
                        send({"type": "accepted", "request_id": identity})
                    except (ValueError, TypeError, OverflowError) as error:
                        send({"type": "error", "request_id": identity, "message": str(error)})
                elif kind in ("read", "cancel") and set(message) == {"type", "request_id"}:
                    handle = handles.get(identity)
                    if kind == "cancel":
                        if handle is None:
                            send({"type": "cancelled", "request_id": identity, "event": None})
                        else:
                            handle.cancel()
                            cancelling.add(identity)
                    elif handle is None:
                        send(
                            {
                                "type": "error",
                                "request_id": identity,
                                "message": "request is unavailable",
                            }
                        )
                    elif identity in reading:
                        raise ValueError("only one read may be outstanding per request")
                    else:
                        reading.add(identity)
                else:
                    raise ValueError("unknown worker command or fields")
            runtime.engine.tick()
            for identity, handle in tuple(handles.items()):
                if outgoing.full():
                    break
                if identity in cancelling and handle.delivery.finish is not None:
                    send(
                        {
                            "type": "cancelled",
                            "request_id": identity,
                            "event": asdict(handle.delivery.finish),
                        }
                    )
                    del handles[identity]
                    reading.discard(identity)
                    cancelling.remove(identity)
                elif identity in reading and identity not in cancelling:
                    try:
                        event = handle.delivery.take(0)
                    except TimeoutError:
                        continue
                    send(
                        {
                            "type": (
                                "finished"
                                if isinstance(event, Finished)
                                else "progress"
                                if isinstance(event, PrefillProgress)
                                else "tokens"
                            ),
                            "request_id": identity,
                            "event": asdict(event),
                        }
                    )
                    reading.remove(identity)
                    if isinstance(event, Finished):
                        del handles[identity]
            if (
                runtime.engine.last_service is None
                and incoming.empty()
                and not runtime.engine.wake.is_set()
            ):
                wake.wait(0.05)
        if failures:
            raise BaseExceptionGroup("private worker transport failed", failures)
        return 0
    except BaseException as error:
        try:
            send({"type": "fatal", "message": f"{type(error).__name__}: {error}"})
        except queue.Full:
            pass
        print(f"worker failed: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop.set()
        lifetime.close()
        try:
            outgoing.put(None, timeout=0.5)
            writer.join(timeout=0.5)
        except queue.Full:
            pass
        retired.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-runtime", action="store_true", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if args.parent_pid <= 1 or os.getppid() != args.parent_pid:
        raise ValueError("worker parent identity does not match its launcher")
    return run(args.parent_pid)


if __name__ == "__main__":
    # Avoid interpreter shutdown waiting on a daemon pipe reader/writer. Owned
    # model resources were already drained by run; the parent observes exact exit.
    try:
        exit_code = main()
    except BaseException as error:
        os.write(2, f"worker startup/cleanup failed: {error}\n".encode())
        exit_code = 1
    os._exit(exit_code)
