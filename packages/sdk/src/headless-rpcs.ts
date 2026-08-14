import { Rpc, RpcGroup } from "@effect/rpc"
import {
  AcnSubscriptionPayload,
  CreateSession,
  GetSession,
  ResyncDisplayView,
  StreamDisplayView,
  StreamEvent,
} from "@magnitudedev/acn-protocol"
import { Effect, ParseResult, Schema } from "effect"

const strictExternalSchema = <A, I, R>(
  schema: Schema.Schema<A, I, R>,
): Schema.Schema<A, unknown, R> => Schema.transformOrFail(
  Schema.Unknown,
  Schema.typeSchema(schema),
  {
    decode: (input) => ParseResult.decodeUnknown(schema, { onExcessProperty: "error" })(input),
    encode: (value) => Effect.succeed(value),
  },
)

const StrictCreateSession = CreateSession
  .setSuccess(strictExternalSchema(CreateSession.successSchema))
  .setError(strictExternalSchema(CreateSession.errorSchema))
const StrictGetSession = GetSession
  .setSuccess(strictExternalSchema(GetSession.successSchema))
  .setError(strictExternalSchema(GetSession.errorSchema))
const StrictResyncDisplayView = ResyncDisplayView
  .setSuccess(strictExternalSchema(ResyncDisplayView.successSchema))
  .setError(strictExternalSchema(ResyncDisplayView.errorSchema))
const StrictStreamDisplayView = Rpc.make(StreamDisplayView._tag, {
  payload: StreamDisplayView.payloadSchema,
  success: strictExternalSchema(AcnSubscriptionPayload(StreamEvent)),
  error: strictExternalSchema(StreamDisplayView.errorSchema),
  stream: true,
})

/**
 * Headless RPCs use the same wire tags and payloads as MagnitudeRpcs, but
 * external success values fail closed on every unknown property before the
 * generic Effect RPC decoder can normalize them.
 */
export const HeadlessRpcs = RpcGroup.make(
  StrictCreateSession,
  StrictGetSession,
  StrictResyncDisplayView,
  StrictStreamDisplayView,
)
