import * as AtomResult from "@effect-atom/atom/Result"
import * as Brand from "effect/Brand"
import type * as Equal from "effect/Equal"
import * as Option from "effect/Option"

export const QueryDefinitionTypeId: unique symbol = Symbol.for(
  "@magnitudedev/effect-query/QueryDefinition"
)

export const MutationDefinitionTypeId: unique symbol = Symbol.for(
  "@magnitudedev/effect-query/MutationDefinition"
)

export interface QueryDefinition {
  readonly [QueryDefinitionTypeId]: true
  readonly name: string
}

export interface MutationDefinition {
  readonly [MutationDefinitionTypeId]: true
  readonly name: string
}

export type QueryKeyPrimitive = string | number | bigint | boolean | symbol | null | undefined

export type QueryKey =
  | QueryKeyPrimitive
  | Equal.Equal
  | ReadonlyArray<QueryKey>
  | { readonly [field: string]: QueryKey }

export interface QueryEntryState {
  readonly fetchStatus: "idle" | "fetching" | "paused"
  readonly isStale: boolean
}

export interface QueryMetadata {
  readonly definition: QueryDefinition
  readonly name: string
  readonly key: QueryKey
  readonly state: QueryEntryState
}

export interface QueryFilter {
  readonly definition?: QueryDefinition
  readonly key?: QueryKey
  readonly exact?: boolean
  readonly stale?: boolean
  readonly fetchStatus?: QueryEntryState["fetchStatus"]
  readonly predicate?: (entry: QueryMetadata) => boolean
}

export type MutationStateId = string & Brand.Brand<"MutationStateId">
export const MutationStateId = Brand.nominal<MutationStateId>()

export type MutationScope = string & Brand.Brand<"MutationScope">
export const MutationScope = Brand.nominal<MutationScope>()

export interface MutationState<Input, Output, Error> {
  readonly id: MutationStateId
  readonly mutation: MutationDefinition
  readonly input: Input
  readonly result: AtomResult.Result<Output, Error>
  readonly scope: Option.Option<MutationScope>
  readonly submittedAt: number
  readonly settledAt: Option.Option<number>
}

/** Mutation state after crossing the heterogeneous client-history boundary. */
export type AnyMutationState = MutationState<unknown, unknown, unknown>

export interface MutationFilter {
  readonly mutation?: MutationDefinition
  readonly scope?: MutationScope
  readonly status?: "pending" | "success" | "error"
  readonly predicate?: (state: AnyMutationState) => boolean
}

export type QueryClientEvent =
  | { readonly _tag: "QueryCreated"; readonly name: string; readonly keyHash: number }
  | { readonly _tag: "QueryRemoved"; readonly name: string; readonly keyHash: number }
  | { readonly _tag: "FetchStarted"; readonly name: string; readonly keyHash: number }
  | { readonly _tag: "FetchSettled"; readonly name: string; readonly keyHash: number; readonly success: boolean }
  | { readonly _tag: "QueryInvalidated"; readonly name: string; readonly keyHash: number }
  | { readonly _tag: "MutationStarted"; readonly name: string; readonly id: MutationStateId }
  | {
    readonly _tag: "MutationSettled"
    readonly name: string
    readonly id: MutationStateId
    readonly success: boolean
  }
