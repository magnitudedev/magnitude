import { Context, Effect, Layer, Option, Redacted } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"

export type HuggingFaceCredentialError = Effect.Effect.Error<ReturnType<IcnClientService["huggingFace"]["configureHuggingFaceCredential"]>>

/**
 * The Hugging Face token ICN presents on Hub requests. `configure` replaces it (`None` makes ICN
 * anonymous) and applies at once, without restarting ICN; ICN never returns it.
 */
export interface IcnHuggingFaceCredentialService {
  readonly configure: (token: Option.Option<Redacted.Redacted<string>>) => Effect.Effect<void, HuggingFaceCredentialError>
}
export class IcnHuggingFaceCredential extends Context.Tag("@magnitudedev/icn/IcnHuggingFaceCredential")<
  IcnHuggingFaceCredential,
  IcnHuggingFaceCredentialService
>() {}

export const makeIcnHuggingFaceCredential = (): Layer.Layer<IcnHuggingFaceCredential, never, IcnClient> =>
  Layer.effect(IcnHuggingFaceCredential, Effect.map(IcnClient, (client) => IcnHuggingFaceCredential.of({
    configure: (token) => client.huggingFace.configureHuggingFaceCredential({
      payload: { token: Option.map(token, Redacted.value) },
    }).pipe(Effect.asVoid),
  })))
