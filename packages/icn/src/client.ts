import * as HttpClient from "@effect/platform/HttpClient"
import { Context, Effect, Layer } from "effect"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import { IcnProcess } from "./lifecycle/index.js"

export type IcnClientService = Effect.Effect.Success<ReturnType<typeof makeIcnApiClient>>

export class IcnClient extends Context.Tag("@magnitudedev/icn/IcnClient")<
  IcnClient,
  IcnClientService
>() {}

export const makeIcnClient = (): Layer.Layer<
  IcnClient,
  never,
  IcnProcess | HttpClient.HttpClient
> =>
  Layer.effect(
    IcnClient,
    Effect.gen(function* () {
      const process = yield* IcnProcess
      return yield* makeIcnApiClient(process.clientOptions)
    }),
  )
