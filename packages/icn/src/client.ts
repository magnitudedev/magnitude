import * as HttpApiClient from "@effect/platform/HttpApiClient";
import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import type * as HttpClientResponse from "@effect/platform/HttpClientResponse";
import { Data, Effect, Option, Schema, Stream } from "effect";
import {
  ApplyTemplateRequest,
  type ApplyTemplateRequestEncoded,
  type ApplyTemplateResponse,
  ChatCompletionChunk,
  type ChatCompletionChunk as ChatCompletionChunkType,
  ChatCompletionRequest,
  type ChatCompletionRequestEncoded,
  ErrorResponse,
  type HealthResponse,
  IcnApi,
  type ModelList,
  type PropsResponse,
} from "./generated/index.js";
import { SseParser, decodeSseJson } from "./stream.js";

export const IcnClientOperation = Schema.Literal(
  "health",
  "list-models",
  "properties",
  "apply-template",
  "chat-completion"
);
export type IcnClientOperation = typeof IcnClientOperation.Type;

export const IcnClientFailureReason = Schema.Literal(
  "invalid-request",
  "transport",
  "remote",
  "invalid-response",
  "incomplete-stream"
);
export type IcnClientFailureReason = typeof IcnClientFailureReason.Type;

export class IcnClientError extends Data.TaggedError("IcnClientError")<{
  readonly operation: IcnClientOperation;
  readonly reason: IcnClientFailureReason;
  readonly status: Option.Option<number>;
  readonly message: string;
}> {}

export interface IcnClient {
  readonly origin: URL;
  readonly health: Effect.Effect<HealthResponse, IcnClientError>;
  readonly listModels: Effect.Effect<ModelList, IcnClientError>;
  readonly properties: Effect.Effect<PropsResponse, IcnClientError>;
  readonly applyTemplate: (
    request: ApplyTemplateRequestEncoded
  ) => Effect.Effect<ApplyTemplateResponse, IcnClientError>;
  readonly chatCompletion: (
    request: ChatCompletionRequestEncoded
  ) => Stream.Stream<ChatCompletionChunkType, IcnClientError>;
}

const clientError = (
  operation: IcnClientOperation,
  reason: IcnClientFailureReason,
  message: string,
  status = Option.none<number>()
) => new IcnClientError({ operation, reason, status, message });

const mapGeneratedFailure = <A, E, R>(
  operation: IcnClientOperation,
  effect: Effect.Effect<A, E, R>
): Effect.Effect<A, IcnClientError, R> =>
  effect.pipe(
    Effect.mapError((error) =>
      Schema.is(ErrorResponse)(error)
        ? clientError(operation, "remote", error.error.message)
        : clientError(operation, "transport", String(error))
    )
  );

const decodeInput = <A, I>(
  operation: IcnClientOperation,
  schema: Schema.Schema<A, I>,
  input: I
): Effect.Effect<A, IcnClientError> =>
  Schema.decode(schema)(input).pipe(
    Effect.mapError((error) =>
      clientError(operation, "invalid-request", String(error))
    )
  );

const remoteResponseError = (
  operation: IcnClientOperation,
  response: HttpClientResponse.HttpClientResponse
): Effect.Effect<never, IcnClientError> =>
  response.json.pipe(
    Effect.flatMap(Schema.decodeUnknown(ErrorResponse)),
    Effect.mapError(() =>
      clientError(
        operation,
        "invalid-response",
        `ICN returned HTTP ${response.status} with an invalid error body`,
        Option.some(response.status)
      )
    ),
    Effect.flatMap(({ error }) =>
      Effect.fail(
        clientError(
          operation,
          "remote",
          error.message,
          Option.some(response.status)
        )
      )
    )
  );

const responseEvents = (
  response: HttpClientResponse.HttpClientResponse
): Stream.Stream<ChatCompletionChunkType, IcnClientError> => {
  const parser = new SseParser();
  let terminated = false;
  const framed = response.stream.pipe(
    Stream.decodeText(),
    Stream.mapConcat((text) => parser.push(text)),
    Stream.concat(
      Stream.sync(() => parser.finish()).pipe(Stream.flattenIterables)
    ),
    Stream.mapError((error) =>
      clientError("chat-completion", "transport", String(error))
    ),
    Stream.takeUntil(({ data }) => {
      if (data !== "[DONE]") return false;
      terminated = true;
      return true;
    }),
    Stream.filter(({ data }) => data !== "[DONE]")
  );
  const decoded = framed.pipe(
    Stream.mapEffect((event) =>
      decodeSseJson(ChatCompletionChunk)(event).pipe(
        Effect.mapError((error) =>
          clientError("chat-completion", "invalid-response", String(error))
        ),
        Effect.flatMap((chunk) =>
          Option.match(chunk.error, {
            onNone: () => Effect.succeed(chunk),
            onSome: (error) =>
              Effect.fail(
                clientError("chat-completion", "remote", error.message)
              ),
          })
        )
      )
    )
  );
  const verifyTermination = Stream.fromEffect(
    Effect.suspend(() =>
      terminated
        ? Effect.void
        : Effect.fail(
            clientError(
              "chat-completion",
              "incomplete-stream",
              "ICN completion stream ended without the [DONE] sentinel"
            )
          )
    )
  ).pipe(Stream.drain);
  return decoded.pipe(Stream.concat(verifyTermination));
};

export const makeIcnClient = (
  origin: URL
): Effect.Effect<IcnClient, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const http = yield* HttpClient.HttpClient;
    const generated = yield* HttpApiClient.makeWith(IcnApi, {
      baseUrl: origin,
      httpClient: http,
    });

    const applyTemplate = (request: ApplyTemplateRequestEncoded) =>
      decodeInput("apply-template", ApplyTemplateRequest, request).pipe(
        Effect.flatMap((payload) =>
          mapGeneratedFailure(
            "apply-template",
            generated.chat.applyChatTemplate({ payload })
          )
        )
      );

    const chatCompletion = (request: ChatCompletionRequestEncoded) =>
      Stream.unwrap(
        Effect.gen(function* () {
          const payload = yield* decodeInput(
            "chat-completion",
            ChatCompletionRequest,
            request
          );
          const httpRequest = yield* HttpClientRequest.post(
            new URL("/v1/chat/completions", origin)
          ).pipe(
            HttpClientRequest.schemaBodyJson(ChatCompletionRequest)(payload),
            Effect.mapError((error) =>
              clientError("chat-completion", "invalid-request", String(error))
            )
          );
          const response = yield* http
            .execute(httpRequest)
            .pipe(
              Effect.mapError((error) =>
                clientError("chat-completion", "transport", String(error))
              )
            );
          if (response.status !== 200) {
            return yield* remoteResponseError("chat-completion", response);
          }
          return responseEvents(response);
        })
      );

    return {
      origin,
      health: mapGeneratedFailure("health", generated.system.health({})),
      listModels: mapGeneratedFailure(
        "list-models",
        generated.models.listModels({})
      ),
      properties: mapGeneratedFailure(
        "properties",
        generated.models.getModelProperties({})
      ),
      applyTemplate,
      chatCompletion,
    };
  });
