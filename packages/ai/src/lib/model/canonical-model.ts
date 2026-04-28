export type ModelId = string & { readonly __brand: "ModelId" }

export interface Model {
  readonly id: ModelId
  readonly name: string
  readonly family: string
  readonly hfRepo: string
  readonly template: string
}
