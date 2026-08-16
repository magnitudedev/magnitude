import { Context } from "effect"
import type { Client } from "@magnitudedev/effect-query"
import type { AcnRpcClientTag } from "@magnitudedev/sdk"

export interface ClientEffectQuery
  extends Pick<Client.Client<AcnRpcClientTag, never>, "query" | "mutation"> {}

/** Connection-local Effect Query materialization used only while building domain services. */
export const ClientEffectQuery = Context.GenericTag<ClientEffectQuery>("client/EffectQuery")
