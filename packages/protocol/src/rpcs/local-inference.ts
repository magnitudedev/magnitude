import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import {
  LocalInferenceOnboardingSnapshot,
  LocalModelDownloadWireEvent,
} from "../schemas/local-inference"

export const GetLocalInferenceOnboardingSnapshot = Rpc.make(
  "GetLocalInferenceOnboardingSnapshot",
  {
    payload: Schema.Struct({}),
    success: LocalInferenceOnboardingSnapshot,
    error: SessionError,
  },
)

export const StartLocalModelDownload = Rpc.make("StartLocalModelDownload", {
  payload: Schema.Struct({ configurationId: Schema.String }),
  success: Schema.Struct({ operationId: Schema.String }),
  error: SessionError,
})

export const SubscribeLocalModelDownload = Rpc.make("SubscribeLocalModelDownload", {
  payload: Schema.Struct({ operationId: Schema.String }),
  success: LocalModelDownloadWireEvent,
  error: SessionError,
  stream: true,
})

export const CancelLocalModelDownload = Rpc.make("CancelLocalModelDownload", {
  payload: Schema.Struct({ operationId: Schema.String }),
  success: Schema.Struct({}),
  error: SessionError,
})

export const ActivateLocalModel = Rpc.make("ActivateLocalModel", {
  payload: Schema.Struct({ selectionId: Schema.String }),
  success: Schema.Struct({
    providerId: Schema.String,
    providerModelId: Schema.String,
    contextTokens: Schema.Number.pipe(Schema.int(), Schema.positive()),
  }),
  error: SessionError,
})

export const CompleteCliModelSetupOnboarding = Rpc.make(
  "CompleteCliModelSetupOnboarding",
  {
    payload: Schema.Struct({}),
    success: Schema.Struct({}),
    error: SessionError,
  },
)
