import { Data, Effect, Option, ParseResult, Schema } from "effect";

export class SseEvent extends Data.Class<{
  readonly data: string;
  readonly event: Option.Option<string>;
  readonly id: Option.Option<string>;
  readonly retry: Option.Option<number>;
}> {}

export class SseDecodeError extends Data.TaggedError("SseDecodeError")<{
  readonly event: SseEvent;
  readonly cause: ParseResult.ParseError;
}> {}

/** Stateful, chunk-boundary-independent SSE framer. One instance belongs to one response. */
export class SseParser {
  private buffer = "";

  push(chunk: string): ReadonlyArray<SseEvent> {
    this.buffer += chunk;
    const events: Array<SseEvent> = [];
    let boundary = findBoundary(this.buffer);
    while (Option.isSome(boundary)) {
      const block = this.buffer.slice(0, boundary.value.start);
      this.buffer = this.buffer.slice(boundary.value.end);
      const event = parseBlock(block);
      if (Option.isSome(event)) events.push(event.value);
      boundary = findBoundary(this.buffer);
    }
    return events;
  }
}

const findBoundary = (
  input: string
): Option.Option<{ readonly start: number; readonly end: number }> => {
  const match = /\r\n\r\n|\n\n|\r\r/.exec(input);
  return Option.fromNullable(match).pipe(
    Option.map((value) => ({
      start: value.index,
      end: value.index + value[0].length,
    }))
  );
};

const parseBlock = (block: string): Option.Option<SseEvent> => {
  const data: Array<string> = [];
  let event = Option.none<string>();
  let id = Option.none<string>();
  let retry = Option.none<number>();

  for (const line of block.split(/\r\n|\r|\n/)) {
    if (line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    const rawValue = separator < 0 ? "" : line.slice(separator + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;
    switch (field) {
      case "data":
        data.push(value);
        break;
      case "event":
        event = Option.some(value);
        break;
      case "id":
        if (!value.includes("\0")) id = Option.some(value);
        break;
      case "retry":
        if (/^[0-9]+$/.test(value)) retry = Option.some(Number(value));
        break;
    }
  }

  return data.length === 0
    ? Option.none()
    : Option.some(new SseEvent({ data: data.join("\n"), event, id, retry }));
};

export const decodeSseJson = <A, I, R>(
  schema: Schema.Schema<A, I, R>
) =>
  (event: SseEvent): Effect.Effect<A, SseDecodeError, R> =>
    Schema.decodeUnknown(Schema.parseJson(schema))(event.data).pipe(
      Effect.mapError((cause) => new SseDecodeError({ event, cause }))
    );
