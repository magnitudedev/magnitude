/**
 * NativeModelResolver — Context.Tag for resolving NativeBoundModels.
 *
 * This is a thin wrapper introduced in Phase 5 as the contract between
 * Cortex and the provider layer.  Phase 6 provides a full implementation
 * that reads from the catalog + ProtocolBindings.
 *
 * Until Phase 6 is complete, the stub implementation returns
 * `Effect.fail(new NativeModelNotConfigured({ slot }))` so that the native
 * cortex path fails gracefully and the system can still run on the xml-act
 * path if needed.
 */

import { Context, Effect, Layer, Schema } from 'effect'
import type { NativeBoundModel } from './native-bound-model'

// =============================================================================
// Errors
// =============================================================================

export class NativeModelNotConfigured extends Schema.TaggedError<NativeModelNotConfigured>()(
  'NativeModelNotConfigured',
  { slot: Schema.String },
) {}

// =============================================================================
// Service shape
// =============================================================================

export interface NativeModelResolverShape {
  readonly resolve: (
    slot: string,
  ) => Effect.Effect<NativeBoundModel, NativeModelNotConfigured>
}

export class NativeModelResolver extends Context.Tag('NativeModelResolver')<
  NativeModelResolver,
  NativeModelResolverShape
>() {}

