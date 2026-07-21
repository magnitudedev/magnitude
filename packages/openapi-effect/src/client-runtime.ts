import * as HttpClient from "@effect/platform/HttpClient";
import type * as HttpBody from "@effect/platform/HttpBody";
import type * as HttpClientError from "@effect/platform/HttpClientError";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import type * as HttpClientResponse from "@effect/platform/HttpClientResponse";
import { Data, Effect, Option, ParseResult, Schema, Stream } from "effect";

type AnySchema = Schema.Schema.AnyNoContext;

export interface GeneratedClientOptions {
  readonly baseUrl: string | URL;
  readonly headers?: Readonly<Record<string, string>>;
  readonly maxFrameBytes?: number;
}

export interface GeneratedClientConnection {
  readonly baseUrl: URL;
  readonly headers: Readonly<Record<string, string>>;
  readonly maxFrameBytes: number;
}

interface RequestOperationDescriptor {
  readonly operationId: string;
  readonly method:
    | "GET"
    | "POST"
    | "PUT"
    | "PATCH"
    | "DELETE"
    | "HEAD"
    | "OPTIONS";
  readonly path: string;
  readonly pathParameters?: AnySchema;
  readonly queryParameters?: AnySchema;
  readonly headers?: AnySchema;
  readonly payload?: AnySchema;
  readonly payloadRequired?: boolean;
  readonly payloadMediaType?: string;
  readonly errors: ReadonlyArray<{
    readonly status: number;
    readonly schema: AnySchema;
    readonly mediaType?: string;
  }>;
}

export interface HttpOperationDescriptor extends RequestOperationDescriptor {
  readonly transport: "http";
  readonly successes: ReadonlyArray<{
    readonly status: number;
    readonly schema: AnySchema;
    readonly mediaType?: string;
  }>;
}

export interface StreamOperationDescriptor extends RequestOperationDescriptor {
  readonly transport: "sse" | "ndjson";
  readonly mediaType: string;
  readonly responseStatus: number;
  readonly eventSchema: AnySchema;
  readonly termination:
    | { readonly type: "sentinel"; readonly value: string }
    | { readonly type: "eof" }
    | { readonly type: "long-lived" };
  readonly reconnect:
    | { readonly type: "none" }
    | { readonly type: "last-event-id" };
}

type SchemaType<A> = A extends AnySchema ? Schema.Schema.Type<A> : never;
type ErrorBody<A extends RequestOperationDescriptor> = SchemaType<
  A["errors"][number]["schema"]
>;
type FieldIfPresent<
  A,
  Key extends PropertyKey,
  OutputKey extends PropertyKey
> = Key extends keyof A
  ? A[Key] extends AnySchema
    ? { readonly [K in OutputKey]: SchemaType<A[Key]> }
    : Record<never, never>
  : Record<never, never>;

type PayloadIfPresent<A> = "payload" extends keyof A
  ? A["payload"] extends AnySchema
    ? A extends { readonly payloadRequired: false }
      ? { readonly payload?: SchemaType<A["payload"]> }
      : { readonly payload: SchemaType<A["payload"]> }
    : Record<never, never>
  : Record<never, never>;

export type OperationInput<A extends RequestOperationDescriptor> =
  FieldIfPresent<A, "pathParameters", "path"> &
    FieldIfPresent<A, "queryParameters", "urlParams"> &
    FieldIfPresent<A, "headers", "headers"> &
    PayloadIfPresent<A>;

export type StreamOperationInput<A extends StreamOperationDescriptor> =
  OperationInput<A>;

export type HttpOperationInput<A extends HttpOperationDescriptor> =
  OperationInput<A>;

export type StreamOperationEvent<A extends StreamOperationDescriptor> =
  SchemaType<A["eventSchema"]>;

export interface GeneratedStreamResponse<
  Event,
  Error,
> {
  readonly status: number;
  readonly headers: HttpClientResponse.HttpClientResponse["headers"];
  readonly events: Stream.Stream<Event, Error>;
}

export class GeneratedClientInputError extends Data.TaggedError(
  "GeneratedClientInputError"
)<{
  readonly operationId: string;
  readonly location: "path" | "query" | "headers" | "payload";
  readonly cause: ParseResult.ParseError;
}> {}

export class GeneratedClientTransportError extends Data.TaggedError(
  "GeneratedClientTransportError"
)<{
  readonly operationId: string;
  readonly cause:
    | HttpClientError.HttpClientError
    | HttpBody.HttpBodyError
    | Error;
}> {}

export class GeneratedClientRemoteError<
  Body = unknown
> extends Data.TaggedError("GeneratedClientRemoteError")<{
  readonly operationId: string;
  readonly status: number;
  readonly headers: HttpClientResponse.HttpClientResponse["headers"];
  readonly body: Body;
}> {}

export class GeneratedClientInvalidResponseError extends Data.TaggedError(
  "GeneratedClientInvalidResponseError"
)<{
  readonly operationId: string;
  readonly status: number;
  readonly message: string;
  readonly cause: Option.Option<unknown>;
}> {}

export class GeneratedClientIncompleteStreamError extends Data.TaggedError(
  "GeneratedClientIncompleteStreamError"
)<{
  readonly operationId: string;
  readonly termination: "sentinel" | "long-lived";
}> {}

export type GeneratedClientError<RemoteBody = unknown> =
  | GeneratedClientInputError
  | GeneratedClientTransportError
  | GeneratedClientRemoteError<RemoteBody>
  | GeneratedClientInvalidResponseError
  | GeneratedClientIncompleteStreamError;

export type StreamOperationError<A extends StreamOperationDescriptor> =
  GeneratedClientError<ErrorBody<A>>;

export type HttpOperationSuccess<A extends HttpOperationDescriptor> =
  SchemaType<A["successes"][number]["schema"]>;

export type HttpOperationError<A extends HttpOperationDescriptor> = Exclude<
  GeneratedClientError<ErrorBody<A>>,
  GeneratedClientIncompleteStreamError
>;

export const makeGeneratedClientConnection = (
  options: GeneratedClientOptions
): GeneratedClientConnection => ({
  baseUrl: new URL(options.baseUrl),
  headers: options.headers ?? {},
  maxFrameBytes: options.maxFrameBytes ?? 1024 * 1024,
});

const encode = (
  operationId: string,
  location: GeneratedClientInputError["location"],
  schema: AnySchema,
  value: unknown
): Effect.Effect<unknown, GeneratedClientInputError> =>
  Schema.encodeUnknown(schema)(value).pipe(
    Effect.mapError(
      (cause) => new GeneratedClientInputError({ operationId, location, cause })
    )
  );

const record = (value: unknown): Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : {};

const scalar = (value: unknown): string =>
  typeof value === "string" ? value : String(value);

const requestFor = <A extends RequestOperationDescriptor>(
  operation: A,
  connection: GeneratedClientConnection,
  input: OperationInput<A>
): Effect.Effect<
  HttpClientRequest.HttpClientRequest,
  GeneratedClientInputError | GeneratedClientTransportError
> =>
  Effect.gen(function* () {
    const raw = input as Readonly<Record<string, unknown>>;
    let pathname = operation.path;
    if (operation.pathParameters !== undefined) {
      const encoded = record(
        yield* encode(
          operation.operationId,
          "path",
          operation.pathParameters,
          raw.path
        )
      );
      for (const [name, value] of Object.entries(encoded))
        pathname = pathname.replaceAll(
          `{${name}}`,
          encodeURIComponent(scalar(value))
        );
    }
    if (/\{[^{}]+\}/.test(pathname)) {
      return yield* new GeneratedClientTransportError({
        operationId: operation.operationId,
        cause: new Error(`Unresolved path parameter in ${pathname}`),
      });
    }

    const url = new URL(pathname, connection.baseUrl);
    if (operation.queryParameters !== undefined) {
      const encoded = record(
        yield* encode(
          operation.operationId,
          "query",
          operation.queryParameters,
          raw.urlParams
        )
      );
      for (const [name, value] of Object.entries(encoded)) {
        if (value === undefined || value === null) continue;
        if (Array.isArray(value)) {
          for (const member of value)
            url.searchParams.append(name, scalar(member));
        } else url.searchParams.set(name, scalar(value));
      }
    }

    let request = HttpClientRequest.make(operation.method)(url).pipe(
      HttpClientRequest.setHeaders(connection.headers)
    );
    if (operation.headers !== undefined) {
      const encoded = record(
        yield* encode(
          operation.operationId,
          "headers",
          operation.headers,
          raw.headers
        )
      );
      request = HttpClientRequest.setHeaders(
        request,
        Object.fromEntries(
          Object.entries(encoded)
            .filter((entry) => entry[1] !== undefined && entry[1] !== null)
            .map(([name, value]) => [name, scalar(value)])
        )
      );
    }
    if (operation.payload !== undefined && raw.payload !== undefined) {
      const payload = yield* encode(
        operation.operationId,
        "payload",
        operation.payload,
        raw.payload
      );
      const mediaType = operation.payloadMediaType ?? "application/json";
      if (mediaType === "application/json") {
        request = yield* HttpClientRequest.bodyJson(request, payload).pipe(
          Effect.mapError(
            (cause) =>
              new GeneratedClientTransportError({
                operationId: operation.operationId,
                cause,
              })
          )
        );
      } else if (mediaType === "text/plain") {
        request = HttpClientRequest.bodyText(
          request,
          scalar(payload),
          mediaType
        );
      } else {
        request = HttpClientRequest.bodyUint8Array(
          request,
          payload as Uint8Array,
          mediaType
        );
      }
    }
    return request;
  });

const remoteFailure = <A extends RequestOperationDescriptor>(
  operation: A,
  response: HttpClientResponse.HttpClientResponse
): Effect.Effect<never, GeneratedClientError<ErrorBody<A>>> => {
  const declared = operation.errors.find(
    ({ status }) => status === response.status
  );
  if (declared === undefined)
    return Effect.fail(
      new GeneratedClientInvalidResponseError({
        operationId: operation.operationId,
        status: response.status,
        message: `Undeclared HTTP response status ${response.status}`,
        cause: Option.none(),
      })
    );
  const body =
    declared.mediaType === undefined
      ? Effect.void
      : declared.mediaType === "application/json"
      ? response.json
      : declared.mediaType === "text/plain"
      ? response.text
      : response.arrayBuffer.pipe(Effect.map((value) => new Uint8Array(value)));
  return body.pipe(
    Effect.mapError(
      (cause) =>
        new GeneratedClientInvalidResponseError({
          operationId: operation.operationId,
          status: response.status,
          message: "Unable to read the declared error response",
          cause: Option.some(cause),
        })
    ),
    Effect.flatMap((body) => Schema.decodeUnknown(declared.schema)(body)),
    Effect.mapError((cause) =>
      cause instanceof GeneratedClientInvalidResponseError
        ? cause
        : new GeneratedClientInvalidResponseError({
            operationId: operation.operationId,
            status: response.status,
            message: "The declared error response did not match its schema",
            cause: Option.some(cause),
          })
    ),
    Effect.flatMap((body) =>
      Effect.fail(
        new GeneratedClientRemoteError({
          operationId: operation.operationId,
          status: response.status,
          headers: response.headers,
          body: body as ErrorBody<A>,
        })
      )
    )
  );
};

interface WireEvent {
  readonly data: string;
  readonly id: Option.Option<string>;
}

const utf8Bytes = (value: string): number =>
  new TextEncoder().encode(value).byteLength;

class SseFramer {
  private buffer = "";

  constructor(private readonly maxFrameBytes: number) {}

  push(chunk: string): ReadonlyArray<WireEvent> {
    this.buffer += chunk;
    const events: WireEvent[] = [];
    let match = /\r\n\r\n|\n\n|\r\r/.exec(this.buffer);
    while (match !== null) {
      const block = this.buffer.slice(0, match.index);
      if (utf8Bytes(block) > this.maxFrameBytes)
        throw new Error(`SSE frame exceeds ${this.maxFrameBytes} bytes`);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const event = parseSseBlock(block);
      if (Option.isSome(event)) events.push(event.value);
      match = /\r\n\r\n|\n\n|\r\r/.exec(this.buffer);
    }
    if (utf8Bytes(this.buffer) > this.maxFrameBytes)
      throw new Error(`SSE frame exceeds ${this.maxFrameBytes} bytes`);
    return events;
  }

  finish(): ReadonlyArray<WireEvent> {
    if (this.buffer.length === 0) return [];
    const block = this.buffer;
    this.buffer = "";
    return Option.toArray(parseSseBlock(block));
  }
}

const parseSseBlock = (block: string): Option.Option<WireEvent> => {
  const data: string[] = [];
  let id = Option.none<string>();
  for (const line of block.split(/\r\n|\r|\n/)) {
    if (line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    const rawValue = separator < 0 ? "" : line.slice(separator + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;
    if (field === "data") data.push(value);
    else if (field === "id" && !value.includes("\0")) id = Option.some(value);
  }
  return data.length === 0
    ? Option.none()
    : Option.some({ data: data.join("\n"), id });
};

class NdjsonFramer {
  private buffer = "";

  constructor(private readonly maxFrameBytes: number) {}

  push(chunk: string): ReadonlyArray<WireEvent> {
    this.buffer += chunk;
    const events: WireEvent[] = [];
    while (true) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) break;
      const line = this.buffer.slice(0, newline).replace(/\r$/, "");
      this.buffer = this.buffer.slice(newline + 1);
      if (utf8Bytes(line) > this.maxFrameBytes)
        throw new Error(`NDJSON frame exceeds ${this.maxFrameBytes} bytes`);
      if (line.trim().length > 0)
        events.push({ data: line, id: Option.none() });
    }
    if (utf8Bytes(this.buffer) > this.maxFrameBytes)
      throw new Error(`NDJSON frame exceeds ${this.maxFrameBytes} bytes`);
    return events;
  }

  finish(): ReadonlyArray<WireEvent> {
    const line = this.buffer.replace(/\r$/, "");
    this.buffer = "";
    return line.trim().length === 0 ? [] : [{ data: line, id: Option.none() }];
  }
}

const responseStream = <A extends StreamOperationDescriptor>(
  operation: A,
  connection: GeneratedClientConnection,
  response: HttpClientResponse.HttpClientResponse,
  onEventId: (id: string) => void
): Stream.Stream<StreamOperationEvent<A>, StreamOperationError<A>> => {
  const framer =
    operation.transport === "sse"
      ? new SseFramer(connection.maxFrameBytes)
      : new NdjsonFramer(connection.maxFrameBytes);
  let terminated = false;
  const framed = response.stream.pipe(
    Stream.decodeText(),
    Stream.mapConcat((chunk) => framer.push(chunk)),
    Stream.concat(
      Stream.sync(() => framer.finish()).pipe(Stream.flattenIterables)
    ),
    Stream.mapError(
      (cause) =>
        new GeneratedClientTransportError({
          operationId: operation.operationId,
          cause,
        })
    ),
    Stream.tap(({ id }) =>
      Option.match(id, {
        onNone: () => Effect.void,
        onSome: (value) => Effect.sync(() => onEventId(value)),
      })
    ),
    Stream.takeUntil(({ data }) => {
      if (
        operation.termination.type === "sentinel" &&
        data === operation.termination.value
      ) {
        terminated = true;
        return true;
      }
      return false;
    }),
    Stream.filter(
      ({ data }) =>
        operation.termination.type !== "sentinel" ||
        data !== operation.termination.value
    ),
    Stream.mapEffect(({ data }) =>
      Schema.decodeUnknown(Schema.parseJson(operation.eventSchema))(data).pipe(
        Effect.mapError(
          (cause) =>
            new GeneratedClientInvalidResponseError({
              operationId: operation.operationId,
              status: response.status,
              message: "Stream event did not match the declared schema",
              cause: Option.some(cause),
            })
        )
      )
    )
  );
  const verify = Stream.fromEffect(
    Effect.suspend(() => {
      if (operation.termination.type === "eof" || terminated)
        return Effect.void;
      return Effect.fail(
        new GeneratedClientIncompleteStreamError({
          operationId: operation.operationId,
          termination: operation.termination.type,
        })
      );
    })
  ).pipe(Stream.drain);
  return framed.pipe(Stream.concat(verify)) as Stream.Stream<
    StreamOperationEvent<A>,
    StreamOperationError<A>
  >;
};

export const makeHttpOperation = <A extends HttpOperationDescriptor>(
  http: HttpClient.HttpClient,
  options: GeneratedClientOptions,
  operation: A
): ((
  input: HttpOperationInput<A>
) => Effect.Effect<HttpOperationSuccess<A>, HttpOperationError<A>>) => {
  const connection = makeGeneratedClientConnection(options);
  return (input) =>
    Effect.gen(function* () {
      const request = yield* requestFor(operation, connection, input);
      const response = yield* http.execute(request).pipe(
        Effect.mapError(
          (cause) =>
            new GeneratedClientTransportError({
              operationId: operation.operationId,
              cause,
            })
        )
      );
      const success = operation.successes.find(
        ({ status }) => status === response.status
      );
      if (success === undefined)
        return yield* remoteFailure(operation, response);
      const body =
        success.mediaType === undefined
          ? Effect.void
          : success.mediaType === "application/json"
          ? response.json
          : success.mediaType === "text/plain"
          ? response.text
          : response.arrayBuffer.pipe(
              Effect.map((value) => new Uint8Array(value))
            );
      const value = yield* body.pipe(
        Effect.mapError(
          (cause) =>
            new GeneratedClientInvalidResponseError({
              operationId: operation.operationId,
              status: response.status,
              message: "Unable to read the declared success response",
              cause: Option.some(cause),
            })
        )
      );
      return yield* Schema.decodeUnknown(success.schema)(value).pipe(
        Effect.mapError(
          (cause) =>
            new GeneratedClientInvalidResponseError({
              operationId: operation.operationId,
              status: response.status,
              message: "The declared success response did not match its schema",
              cause: Option.some(cause),
            })
        )
      );
    }) as Effect.Effect<HttpOperationSuccess<A>, HttpOperationError<A>>;
};

export const makeStreamOperation = <A extends StreamOperationDescriptor>(
  http: HttpClient.HttpClient,
  options: GeneratedClientOptions,
  operation: A
): ((
  input: StreamOperationInput<A>
) => Effect.Effect<
  GeneratedStreamResponse<StreamOperationEvent<A>, StreamOperationError<A>>,
  StreamOperationError<A>
>) => {
  const connection = makeGeneratedClientConnection(options);
  return (input) => {
    let lastEventId: string | undefined;
    const admit = (): Effect.Effect<
      GeneratedStreamResponse<StreamOperationEvent<A>, StreamOperationError<A>>,
      StreamOperationError<A>
    > => Effect.gen(function* () {
      let request = yield* requestFor(operation, connection, input);
      if (
        operation.reconnect.type === "last-event-id" &&
        lastEventId !== undefined
      )
        request = HttpClientRequest.setHeader(
          request,
          "last-event-id",
          lastEventId
        );
      const response = yield* http.execute(request).pipe(
        Effect.mapError(
          (cause) =>
            new GeneratedClientTransportError({
              operationId: operation.operationId,
              cause,
            })
        )
      );
      if (response.status !== operation.responseStatus)
        return yield* remoteFailure(operation, response);
      const first = responseStream(
        operation,
        connection,
        response,
        (id) => (lastEventId = id)
      );
      const reconnect = (events: Stream.Stream<StreamOperationEvent<A>, StreamOperationError<A>>): Stream.Stream<
        StreamOperationEvent<A>,
        StreamOperationError<A>
      > => events.pipe(
        Stream.catchAll((error) =>
          operation.reconnect.type === "last-event-id" &&
          (error instanceof GeneratedClientIncompleteStreamError ||
            error instanceof GeneratedClientTransportError)
            ? Stream.fromEffect(Effect.sleep("100 millis")).pipe(
                Stream.drain,
                Stream.concat(Stream.unwrap(admit().pipe(
                  Effect.map((next) => reconnect(next.events)),
                )))
              )
            : Stream.fail(error)
        )
      );
      return {
        status: response.status,
        headers: response.headers,
        events: reconnect(first),
      };
    });
    return admit();
  };
};
