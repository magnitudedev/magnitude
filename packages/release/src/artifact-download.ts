import { createHash } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import * as Path from "@effect/platform/Path"
import {
  Cause,
  Clock,
  Data,
  Duration,
  Effect,
  Either,
  Exit,
  Option,
  Random,
  Ref,
  Sink,
  Stream,
} from "effect"
import type { ArtifactByteProgress } from "./installation-progress"

export type ArtifactDownloadStrategy =
  | { readonly _tag: "Sequential" }
  | {
    readonly _tag: "Segmented"
    readonly concurrency: number
    readonly chunkBytes: number
    readonly fallbackToSequential: boolean
  }

export interface ArtifactDownloadProgress {
  readonly strategy: ArtifactDownloadStrategy["_tag"]
  readonly acceptedBytes: number
  readonly totalBytes: number
  readonly attempt: number
}

/**
 * A transient failure is retried with jittered exponential backoff from `retryDelay` up to
 * `retryMaxDelay`, until `retryWindow` passes without the transfer accepting a byte.
 */
export interface ArtifactDownloadPolicy {
  readonly retryDelay: Duration.DurationInput
  readonly retryMaxDelay: Duration.DurationInput
  readonly retryWindow: Duration.DurationInput
  readonly attemptTimeout: Duration.DurationInput
  readonly stallTimeout: Duration.DurationInput
  readonly totalTimeout: Duration.DurationInput
}

export const defaultArtifactDownloadPolicy: ArtifactDownloadPolicy = {
  retryDelay: "1 second",
  retryMaxDelay: "30 seconds",
  retryWindow: "10 minutes",
  attemptTimeout: "10 minutes",
  stallTimeout: "60 seconds",
  totalTimeout: "31 minutes",
}

export interface ArtifactDownloadInput {
  readonly url: string
  readonly destination: string
  readonly bytes: number
  readonly sha256: string
  readonly strategy: ArtifactDownloadStrategy
  readonly policy: ArtifactDownloadPolicy
  readonly onProgress: Option.Option<
    (progress: ArtifactDownloadProgress) => Effect.Effect<void>
  >
  readonly onVerificationProgress: Option.Option<
    (progress: ArtifactByteProgress) => Effect.Effect<void>
  >
}

export interface ArtifactDownloadResult {
  readonly destination: string
  readonly strategy: ArtifactDownloadStrategy["_tag"]
  readonly bytes: number
  readonly sha256: string
}

/**
 * Parts of an unfinished download live beside its destination, named by the artifact's digest, so
 * a later download of the same artifact resumes them. One process downloads an artifact at a time.
 */
export const partialDirectoryName = (sha256: string) => `.partial-${sha256}`

export class ArtifactDownloadError extends Data.TaggedError("ArtifactDownloadError")<{
  readonly phase: "request" | "protocol" | "stream" | "filesystem" | "integrity"
  readonly message: string
  readonly transient: boolean
}> {}

class RangeUnsupported extends Data.TaggedError("RangeUnsupported")<{
  readonly message: string
}> {}

interface Range {
  readonly index: number
  readonly start: number
  readonly end: number
}

interface RangeRepresentation {
  readonly etag: Option.Option<string>
}

interface SegmentedProgress {
  readonly acceptedByPart: Ref.Ref<readonly number[]>
  readonly reporting: Effect.Semaphore
}

const downloadError = (
  phase: ArtifactDownloadError["phase"],
  message: string,
  transient = false,
) => new ArtifactDownloadError({ phase, message, transient })

const transientStatus = (status: number) =>
  status === 408 || status === 429 || status >= 500

const parseContentRange = (
  value: string | undefined,
): Effect.Effect<
  { readonly start: number; readonly end: number; readonly total: number },
  ArtifactDownloadError
> => {
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(value ?? "")
  return match
    ? Effect.succeed({
      start: Number(match[1]),
      end: Number(match[2]),
      total: Number(match[3]),
    })
    : Effect.fail(downloadError(
      "protocol",
      `range response has invalid Content-Range ${value ?? "<missing>"}`,
    ))
}

/**
 * An outage of a release host lasts seconds to minutes, so a part keeps retrying for the whole
 * window, and any byte it accepts restarts the window and the backoff.
 */
const retryTransfer = <A, E, R>(
  policy: ArtifactDownloadPolicy,
  retriable: (error: E) => boolean,
  accepted: Effect.Effect<number>,
  transfer: Effect.Effect<A, E, R>,
): Effect.Effect<A, E, R> =>
  Effect.gen(function* () {
    const initialDelay = Duration.toMillis(policy.retryDelay)
    const maximumDelay = Duration.toMillis(policy.retryMaxDelay)
    const window = Duration.toMillis(policy.retryWindow)
    let delay = initialDelay
    let acceptedBytes = yield* accepted
    let idleSince = yield* Clock.currentTimeMillis
    for (;;) {
      const result = yield* Effect.either(transfer)
      if (Either.isRight(result)) return result.right
      if (!retriable(result.left)) return yield* Effect.fail(result.left)
      const now = yield* Clock.currentTimeMillis
      const currentBytes = yield* accepted
      if (currentBytes > acceptedBytes) {
        acceptedBytes = currentBytes
        idleSince = now
        delay = initialDelay
      }
      if (now - idleSince >= window) return yield* Effect.fail(result.left)
      yield* Effect.sleep(Duration.millis(delay * (0.8 + 0.4 * (yield* Random.next))))
      delay = Math.min(delay * 2, maximumDelay)
    }
  })

const isTransient = (error: ArtifactDownloadError) => error.transient

const report = (
  input: ArtifactDownloadInput,
  strategy: ArtifactDownloadStrategy["_tag"],
  acceptedBytes: number,
  attempt: number,
) =>
  Option.match(input.onProgress, {
    onNone: () => Effect.void,
    onSome: (onProgress) =>
      onProgress({
        strategy,
        acceptedBytes,
        totalBytes: input.bytes,
        attempt,
      }),
  })

const reportVerification = (
  input: ArtifactDownloadInput,
  completedBytes: number,
) =>
  Option.match(input.onVerificationProgress, {
    onNone: () => Effect.void,
    onSome: (onProgress) =>
      onProgress({
        completedBytes,
        totalBytes: input.bytes,
      }),
  })

const discardResponse = (response: HttpClientResponse.HttpClientResponse) =>
  response.stream.pipe(
    Stream.take(1),
    Stream.runDrain,
    Effect.ignore,
  )

const streamResponse = (
  input: ArtifactDownloadInput,
  strategy: ArtifactDownloadStrategy["_tag"],
  response: HttpClientResponse.HttpClientResponse,
  destination: string,
  expectedBytes: number,
  attempt: number,
  onAccepted: (bytes: number) => Effect.Effect<number>,
  flag: "w" | "a",
): Effect.Effect<void, ArtifactDownloadError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    let received = 0
    yield* response.stream.pipe(
      Stream.mapError(() =>
        downloadError("stream", "artifact response stream failed", true)
      ),
      Stream.timeoutFail(
        () => downloadError("stream", "artifact response stalled", true),
        input.policy.stallTimeout,
      ),
      Stream.tap((chunk) => {
        received += chunk.byteLength
        if (received > expectedBytes) {
          return Effect.fail(downloadError(
            "protocol",
            "artifact response exceeded its declared range",
          ))
        }
        return onAccepted(chunk.byteLength).pipe(
          Effect.flatMap((acceptedBytes) =>
            report(input, strategy, acceptedBytes, attempt)
          ),
        )
      }),
      Stream.run(
        fs.sink(destination, { flag, mode: 0o600 }).pipe(
          Sink.mapError(() =>
            downloadError("filesystem", "unable to write downloaded artifact")
          ),
        ),
      ),
    )
    if (received !== expectedBytes) {
      return yield* downloadError(
        "protocol",
        `artifact response contained ${received} bytes, expected ${expectedBytes}`,
        true,
      )
    }
  })

const validateCompleteFile = (
  input: ArtifactDownloadInput,
  path: string,
): Effect.Effect<void, ArtifactDownloadError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const info = yield* fs.stat(path).pipe(
      Effect.mapError(() =>
        downloadError("filesystem", "unable to inspect downloaded artifact")
      ),
    )
    if (Number(info.size) !== input.bytes) {
      return yield* downloadError(
        "integrity",
        `downloaded artifact has ${info.size} bytes, expected ${input.bytes}`,
      )
    }
    const digest = createHash("sha256")
    let completedBytes = 0
    yield* fs.stream(path).pipe(
      Stream.tap((chunk) =>
        Effect.sync(() => {
          completedBytes += chunk.byteLength
          digest.update(chunk)
        }).pipe(
          Effect.zipRight(reportVerification(input, completedBytes))
        )
      ),
      Stream.runDrain,
      Effect.mapError(() =>
        downloadError("filesystem", "unable to hash downloaded artifact")
      ),
    )
    const sha256 = digest.digest("hex")
    if (sha256 !== input.sha256) {
      return yield* downloadError(
        "integrity",
        `downloaded artifact SHA-256 ${sha256} differs from ${input.sha256}`,
      )
    }
  })

const sequentialAttempt = (
  input: ArtifactDownloadInput,
  staging: string,
  attempt: number,
  received: Ref.Ref<number>,
): Effect.Effect<
  void,
  ArtifactDownloadError,
  FileSystem.FileSystem | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const accepted = yield* Ref.make(0)
    yield* report(input, "Sequential", 0, attempt)
    const response = yield* client.execute(
      HttpClientRequest.get(input.url).pipe(
        HttpClientRequest.setHeader("accept-encoding", "identity;q=1, *;q=0"),
      ),
    ).pipe(
      Effect.mapError(() =>
        downloadError("request", "artifact request failed", true)
      ),
    )
    if (response.status < 200 || response.status >= 300) {
      yield* discardResponse(response)
      return yield* downloadError(
        "request",
        `artifact endpoint returned HTTP ${response.status}`,
        transientStatus(response.status),
      )
    }
    const declared = Number(response.headers["content-length"])
    if (Number.isFinite(declared) && declared !== input.bytes) {
      return yield* downloadError(
        "protocol",
        `artifact response declares ${declared} bytes, expected ${input.bytes}`,
      )
    }
    yield* streamResponse(
      input,
      "Sequential",
      response,
      staging,
      input.bytes,
      attempt,
      (bytes) =>
        Ref.update(received, (current) => current + bytes).pipe(
          Effect.zipRight(Ref.updateAndGet(accepted, (current) => current + bytes)),
        ),
      "w",
    )
  }).pipe(
    Effect.timeoutFail({
      duration: input.policy.attemptTimeout,
      onTimeout: () =>
        downloadError("stream", "artifact attempt timed out", true),
    }),
  )

const downloadSequential = (
  input: ArtifactDownloadInput,
  staging: string,
): Effect.Effect<
  void,
  ArtifactDownloadError,
  FileSystem.FileSystem | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const attempt = yield* Ref.make(0)
    const received = yield* Ref.make(0)
    yield* retryTransfer(
      input.policy,
      isTransient,
      Ref.get(received),
      Ref.updateAndGet(attempt, (value) => value + 1).pipe(
        Effect.flatMap((currentAttempt) =>
          sequentialAttempt(input, staging, currentAttempt, received)
        ),
      ),
    )
    yield* validateCompleteFile(input, staging)
  })

const rangeRequest = (
  input: ArtifactDownloadInput,
  range: Range,
  representation: Option.Option<string>,
  allowUnsupported: boolean,
): Effect.Effect<
  HttpClientResponse.HttpClientResponse,
  ArtifactDownloadError | RangeUnsupported,
  HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const request = HttpClientRequest.get(input.url).pipe(
      HttpClientRequest.setHeaders({
        range: `bytes=${range.start}-${range.end}`,
        "accept-encoding": "identity;q=1, *;q=0",
        "if-range": Option.getOrUndefined(representation),
      }),
    )
    const response = yield* client.execute(request).pipe(
      Effect.mapError(() =>
        downloadError("request", `range ${range.index} request failed`, true)
      ),
    )
    if (allowUnsupported && response.status === 200) {
      yield* discardResponse(response)
      return yield* new RangeUnsupported({
        message: "artifact endpoint does not support byte ranges",
      })
    }
    if (response.status !== 206) {
      yield* discardResponse(response)
      return yield* downloadError(
        "request",
        `range ${range.index} returned HTTP ${response.status}, expected 206`,
        transientStatus(response.status),
      )
    }
    const contentRange = yield* parseContentRange(
      response.headers["content-range"],
    )
    if (
      contentRange.start !== range.start ||
      contentRange.end !== range.end ||
      contentRange.total !== input.bytes
    ) {
      yield* discardResponse(response)
      return yield* downloadError(
        "protocol",
        `range ${range.index} returned inconsistent Content-Range: ${contentRange.start}-${contentRange.end}/${contentRange.total}; expected ${range.start}-${range.end}/${input.bytes}`,
        true,
      )
    }
    const expectedBytes = range.end - range.start + 1
    const declared = Number(response.headers["content-length"])
    if (Number.isFinite(declared) && declared !== expectedBytes) {
      yield* discardResponse(response)
      return yield* downloadError(
        "protocol",
        `range ${range.index} declares ${declared} bytes, expected ${expectedBytes}`,
      )
    }
    if (
      Option.isSome(representation) &&
      response.headers.etag !== undefined &&
      response.headers.etag !== representation.value
    ) {
      yield* discardResponse(response)
      return yield* downloadError(
        "protocol",
        `range ${range.index} returned a different ETag`,
      )
    }
    return response
  })

const probeRanges = (
  input: ArtifactDownloadInput,
): Effect.Effect<
  RangeRepresentation,
  ArtifactDownloadError | RangeUnsupported,
  HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const response = yield* rangeRequest(
      input,
      { index: 0, start: 0, end: 0 },
      Option.none(),
      true,
    )
    const body = yield* response.arrayBuffer.pipe(
      Effect.mapError(() =>
        downloadError("stream", "range probe response failed", true)
      ),
    )
    if (body.byteLength !== 1) {
      return yield* downloadError(
        "protocol",
        `range probe returned ${body.byteLength} bytes, expected 1`,
      )
    }
    const etag = Option.fromNullable(response.headers.etag).pipe(
      Option.filter((value) => !value.startsWith("W/")),
    )
    return { etag }
  })

const updateSegmentedProgress = (
  input: ArtifactDownloadInput,
  progress: SegmentedProgress,
  rangeIndex: number,
  update: (current: number) => number,
  attempt: number,
): Effect.Effect<void> =>
  progress.reporting.withPermits(1)(
    Ref.modify(progress.acceptedByPart, (parts) => {
      const updated = [...parts]
      updated[rangeIndex] = update(updated[rangeIndex]!)
      return [updated.reduce((sum, bytes) => sum + bytes, 0), updated] as const
    }).pipe(
      Effect.flatMap((acceptedBytes) =>
        report(input, "Segmented", acceptedBytes, attempt)
      ),
    ),
  )

const partBytes = (
  part: string,
  rangeBytes: number,
): Effect.Effect<number, ArtifactDownloadError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const info = yield* fs.stat(part).pipe(Effect.option)
    if (Option.isNone(info)) return 0
    const bytes = Number(info.value.size)
    if (bytes <= rangeBytes) return bytes
    yield* fs.remove(part).pipe(
      Effect.mapError(() =>
        downloadError("filesystem", "unable to discard an oversized range part")
      ),
    )
    return 0
  })

/** A part keeps the bytes it has, so a retry or a later download asks only for the rest of its range. */
const downloadRange = (
  input: ArtifactDownloadInput,
  range: Range,
  representation: RangeRepresentation,
  part: string,
  progress: SegmentedProgress,
): Effect.Effect<
  void,
  ArtifactDownloadError,
  FileSystem.FileSystem | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const attempt = yield* Ref.make(0)
    const received = yield* Ref.make(0)
    const rangeBytes = range.end - range.start + 1
    const once = Effect.gen(function* () {
      const currentAttempt = yield* Ref.updateAndGet(attempt, (value) => value + 1)
      const kept = yield* partBytes(part, rangeBytes)
      yield* updateSegmentedProgress(
        input,
        progress,
        range.index,
        () => kept,
        currentAttempt,
      )
      if (kept === rangeBytes) return
      const response = yield* rangeRequest(
        input,
        { index: range.index, start: range.start + kept, end: range.end },
        representation.etag,
        false,
      ).pipe(
        Effect.catchTag("RangeUnsupported", (error) =>
          Effect.fail(downloadError("protocol", error.message))
        ),
      )
      yield* streamResponse(
        input,
        "Segmented",
        response,
        part,
        rangeBytes - kept,
        currentAttempt,
        (bytes) =>
          Ref.update(received, (current) => current + bytes).pipe(
            Effect.zipRight(progress.reporting.withPermits(1)(
              Ref.modify(progress.acceptedByPart, (parts) => {
                const updated = [...parts]
                updated[range.index] = updated[range.index]! + bytes
                return [
                  updated.reduce((sum, value) => sum + value, 0),
                  updated,
                ] as const
              }),
            )),
          ),
        "a",
      )
    }).pipe(
      Effect.timeoutFail({
        duration: input.policy.attemptTimeout,
        onTimeout: () =>
          downloadError("stream", `range ${range.index} timed out`, true),
      }),
    )
    yield* retryTransfer(input.policy, isTransient, Ref.get(received), once)
  })

const assembleRanges = (
  input: ArtifactDownloadInput,
  ranges: readonly Range[],
  directory: string,
  staging: string,
): Effect.Effect<void, ArtifactDownloadError, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const digest = createHash("sha256")
    let bytes = 0
    for (const range of ranges) {
      const part = path.join(directory, `part-${range.index}`)
      yield* fs.stream(part).pipe(
        Stream.tap((chunk) =>
          Effect.sync(() => {
            bytes += chunk.byteLength
            digest.update(chunk)
          }).pipe(Effect.zipRight(reportVerification(input, bytes)))
        ),
        Stream.run(fs.sink(staging, {
          flag: range.index === 0 ? "w" : "a",
          mode: 0o600,
        })),
        Effect.mapError(() =>
          downloadError("filesystem", `unable to assemble range ${range.index}`)
        ),
      )
      yield* fs.remove(part).pipe(
        Effect.mapError(() =>
          downloadError("filesystem", `unable to remove range ${range.index}`)
        ),
      )
    }
    if (bytes !== input.bytes || digest.digest("hex") !== input.sha256) {
      return yield* downloadError(
        "integrity",
        "assembled artifact digest or size differs from its manifest",
      )
    }
  })

const downloadSegmented = (
  input: ArtifactDownloadInput,
  strategy: Extract<ArtifactDownloadStrategy, { readonly _tag: "Segmented" }>,
  directory: string,
  staging: string,
): Effect.Effect<
  void,
  ArtifactDownloadError | RangeUnsupported,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    if (
      !Number.isSafeInteger(strategy.concurrency) ||
      strategy.concurrency <= 0 ||
      !Number.isSafeInteger(strategy.chunkBytes) ||
      strategy.chunkBytes <= 0
    ) {
      return yield* downloadError(
        "protocol",
        "segmented strategy requires positive integer concurrency and chunkBytes",
      )
    }
    const path = yield* Path.Path
    const probeAttempt = yield* Ref.make(0)
    // Each attempt reports, so an owner watching for a stuck startup sees the download is still trying.
    const representation = yield* retryTransfer(
      input.policy,
      (error) =>
        error._tag === "ArtifactDownloadError" && (error.transient || error.phase === "protocol"),
      Effect.succeed(0),
      Ref.updateAndGet(probeAttempt, (value) => value + 1).pipe(
        Effect.flatMap((attempt) => report(input, "Segmented", 0, attempt)),
        Effect.zipRight(probeRanges(input)),
        Effect.timeoutFail({
          duration: input.policy.attemptTimeout,
          onTimeout: () =>
            downloadError("stream", "range probe timed out", true),
        }),
      ),
    )
    const ranges = Array.from(
      { length: Math.ceil(input.bytes / strategy.chunkBytes) },
      (_, index): Range => ({
        index,
        start: index * strategy.chunkBytes,
        end: Math.min((index + 1) * strategy.chunkBytes, input.bytes) - 1,
      }),
    )
    const progress: SegmentedProgress = {
      acceptedByPart: yield* Ref.make<readonly number[]>(ranges.map(() => 0)),
      reporting: yield* Effect.makeSemaphore(1),
    }
    yield* Effect.forEach(
      ranges,
      (range) =>
        downloadRange(
          input,
          range,
          representation,
          path.join(directory, `part-${range.index}`),
          progress,
        ),
      { concurrency: strategy.concurrency, discard: true },
    )
    yield* assembleRanges(input, ranges, directory, staging)
  })

export const downloadArtifact = (
  input: ArtifactDownloadInput,
): Effect.Effect<
  ArtifactDownloadResult,
  ArtifactDownloadError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
      if (
        !Number.isSafeInteger(input.bytes) ||
        input.bytes <= 0 ||
        !/^[a-f0-9]{64}$/.test(input.sha256)
      ) {
        return yield* downloadError(
          "protocol",
          "artifact bytes and SHA-256 must be valid",
        )
      }
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const parent = path.dirname(input.destination)
      yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 }).pipe(
        Effect.mapError(() =>
          downloadError("filesystem", "unable to create download directory")
        ),
      )
      const exists = yield* fs.exists(input.destination).pipe(
        Effect.mapError(() =>
          downloadError("filesystem", "unable to inspect download destination")
        ),
      )
      if (exists) {
        return yield* downloadError(
          "filesystem",
          "download destination already exists",
        )
      }
      const partial = path.join(parent, partialDirectoryName(input.sha256))
      return yield* Effect.acquireUseRelease(
        fs.makeDirectory(partial, { recursive: true, mode: 0o700 }).pipe(
          Effect.as(partial),
          Effect.mapError(() =>
            downloadError("filesystem", "unable to create download staging directory")
          ),
        ),
        (directory) =>
          Effect.uninterruptibleMask((restore) => {
            const staging = path.join(directory, "artifact")
            const transfer = input.strategy._tag === "Sequential"
              ? downloadSequential(input, staging).pipe(
                Effect.as<ArtifactDownloadStrategy["_tag"]>("Sequential"),
              )
              : downloadSegmented(
                input,
                input.strategy,
                directory,
                staging,
              ).pipe(
                Effect.as<ArtifactDownloadStrategy["_tag"]>("Segmented"),
                Effect.catchTag("RangeUnsupported", (error) =>
                  input.strategy._tag === "Segmented" &&
                    input.strategy.fallbackToSequential
                    ? downloadSequential(input, staging).pipe(
                      Effect.as<ArtifactDownloadStrategy["_tag"]>("Sequential"),
                    )
                    : Effect.fail(downloadError("protocol", error.message))
                ),
              )
            return Effect.gen(function* () {
              const usedStrategy = yield* restore(transfer.pipe(
                Effect.timeoutFail({
                  duration: input.policy.totalTimeout,
                  onTimeout: () =>
                    downloadError(
                      "stream",
                      "artifact acquisition exceeded its total deadline",
                      true,
                    ),
                }),
              ))
              yield* fs.rename(staging, input.destination).pipe(
                Effect.mapError(() =>
                  downloadError(
                    "filesystem",
                    "unable to publish downloaded artifact",
                  )
                ),
              )
              return {
                destination: input.destination,
                strategy: usedStrategy,
                bytes: input.bytes,
                sha256: input.sha256,
              }
            })
          }),
        // An interrupted or transiently failed transfer keeps its parts for the next download to resume.
        (directory, exit) =>
          Exit.isFailure(exit) && Exit.match(exit, {
            onSuccess: () => false,
            onFailure: (cause) => Option.match(Cause.failureOption(cause), {
              onNone: () => Cause.isInterruptedOnly(cause),
              onSome: (error) => error.transient,
            }),
          })
            ? Effect.void
            : fs.remove(directory, { recursive: true, force: true }).pipe(
              Effect.catchAll(() => Effect.void),
            ),
      )
    })
