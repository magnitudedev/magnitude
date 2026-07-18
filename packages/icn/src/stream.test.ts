import { Effect, Option, Schema } from "effect";
import { describe, expect, it } from "vitest";
import { decodeSseJson, SseParser } from "./stream.js";

describe("SseParser", () => {
  it("handles fragmented CRLF, comments, multiline data, and metadata", () => {
    const parser = new SseParser();
    expect(
      parser.push(': keepalive\r\nid: 42\r\nevent: delta\r\ndata: {"a":')
    ).toEqual([]);
    const events = parser.push("1}\r\ndata: tail\r\nretry: 500\r\n\r\n");
    expect(events).toHaveLength(1);
    expect(events[0]?.data).toBe('{"a":1}\ntail');
    expect(Option.getOrNull(events[0]!.id)).toBe("42");
    expect(Option.getOrNull(events[0]!.event)).toBe("delta");
    expect(Option.getOrNull(events[0]!.retry)).toBe(500);
  });

  it("decodes each JSON data value through Effect Schema", async () => {
    const parser = new SseParser();
    const [event] = parser.push('data: {"value":1}\n\n');
    const value = await Effect.runPromise(
      decodeSseJson(Schema.Struct({ value: Schema.Number }))(event!)
    );
    expect(value).toEqual({ value: 1 });
  });

  it("dispatches a final event when EOF arrives without a blank line", () => {
    const parser = new SseParser();
    expect(parser.push("data: final")).toEqual([]);
    expect(parser.finish().map(({ data }) => data)).toEqual(["final"]);
    expect(parser.finish()).toEqual([]);
  });
});
