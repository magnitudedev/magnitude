/**
 * Image Query Tool
 *
 * Sends an image file to the opposite slot's vision-capable model with an
 * optional query and returns a text description.
 */

import { Context, Effect, Option, Schema, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { defineHarnessTool, StreamValidationError } from '@magnitudedev/harness'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { readImageFileForModel } from '../util/read-image-file'
import { Fs, resolveFsPath } from '../services/fs'
import { formatStreamFailureMessage, Prompt, type ModelStreamTerminal } from '@magnitudedev/ai'
import { AgentModelResolver } from '../model/model-resolver'
import { IMAGE_DESCRIPTION_PROMPT } from '../util/image-prompts'
import type { SlotConfig } from '../ambient/config-ambient'

export interface ImageQueryTargetService {
  readonly slot: SlotConfig | null
}

/** Captured opposite-slot selection for this turn. */
export class ImageQueryTarget extends Context.Tag('ImageQueryTarget')<ImageQueryTarget, ImageQueryTargetService>() {}

const ImageQueryErrorSchema = Schema.Struct({ message: Schema.String })

// =============================================================================
// Error helper
// =============================================================================

type ImageQueryError = { readonly _tag: 'ImageQueryError'; readonly message: string }

function imageError(message: string): ImageQueryError {
  return { _tag: 'ImageQueryError', message }
}

function streamTerminalErrorMessage(terminal: ModelStreamTerminal): Option.Option<string> {
  switch (terminal._tag) {
    case 'StreamCompleted':
      return Option.none()
    case 'StreamFailed':
      return Option.some(formatStreamFailureMessage(terminal.cause))
  }
}

// =============================================================================
// queryImage
// =============================================================================

export const queryImageTool = defineHarnessTool({
  definition: {
    name: 'query_image',
    description: 'Query an image file through the other configured model when the active model does not support vision. Supports PNG, JPEG, WebP, GIF, and SVG files. When no query is provided, a detailed description of the image is returned.',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({
        description: 'Relative path to an image file from cwd. Use $M/ prefix for scratchpad path.'
      }),
      query: Schema.optionalWith(Schema.String, { default: () => IMAGE_DESCRIPTION_PROMPT, exact: true }).annotations({
        description: 'Question or instruction about what to look for in the image. Defaults to a general description request.'
      }),
    }),
    outputSchema: Schema.String,
  },
  errorSchema: ImageQueryErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.path?.isFinal) return {}
      const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
      const fs = yield* Fs
      const fullPath = resolveFsPath(input.path.value, cwd, scratchpadPath)
      const exists = yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (!exists) {
        return yield* new StreamValidationError({ message: `File not found: ${input.path.value}` })
      }
      return {}
    }),
  },
  execute: ({ path: filePath, query }, _ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const fullPath = resolveFsPath(filePath, cwd, scratchpadPath)

    yield* fs.readFile(fullPath).pipe(
      Effect.mapError(() => imageError(`Failed to read image: ${filePath}`))
    )

    const imageResult = yield* Effect.tryPromise({
      try: () => readImageFileForModel(fullPath),
      catch: (e) => imageError(e instanceof Error ? e.message : `Failed to read image: ${filePath}`),
    })

    const modelResolver = yield* AgentModelResolver
    const httpClient = yield* HttpClient.HttpClient
    const target = yield* ImageQueryTarget
    if (!target.slot) return yield* Effect.fail(imageError('The opposite model slot is unavailable'))
    const imageModel = yield* modelResolver.resolveSlotConfig(target.slot).pipe(
      Effect.provideService(HttpClient.HttpClient, httpClient),
    )

    const prompt = Prompt.from({
      messages: [{
        _tag: 'UserMessage',
        parts: [
          { _tag: 'TextPart', text: query },
          { _tag: 'ImagePart', mediaType: imageResult.mediaType, data: imageResult.base64 },
        ],
      }],
    })

    const streamResult = yield* imageModel.model.stream(prompt, [], { maxTokens: 2048 }).pipe(
      Effect.mapError((err) => imageError(`Image query failed: ${err.message}`))
    )

    const collected = yield* Stream.runFold(
      streamResult.events,
      { text: '', streamError: Option.none<string>() },
      (state, event) => {
        if (event._tag === 'message_delta') return { ...state, text: state.text + event.text }
        if (event._tag === 'stream_end') {
          return { ...state, streamError: streamTerminalErrorMessage(event.terminal) }
        }
        return state
      },
    )

    if (Option.isSome(collected.streamError)) {
      return yield* Effect.fail(imageError(`Image query stream failed: ${collected.streamError.value}`))
    }

    return collected.text.trim() || `[No response from image model for: ${filePath}]`
  }),
})
