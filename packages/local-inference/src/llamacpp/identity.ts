import { Schema } from "effect"

const id = <const Brand extends string>(brand: Brand, max = 1024) => Schema.String.pipe(Schema.minLength(1), Schema.maxLength(max), Schema.brand(brand))
export const LlamaDistributionVariantId = id("LlamaDistributionVariantId")
export type LlamaDistributionVariantId = Schema.Schema.Type<typeof LlamaDistributionVariantId>
export const LlamaBinaryFingerprint = id("LlamaBinaryFingerprint", 256)
export type LlamaBinaryFingerprint = Schema.Schema.Type<typeof LlamaBinaryFingerprint>
export const LlamaBuildCommitId = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{7,64}$/i), Schema.brand("LlamaBuildCommitId"))
export type LlamaBuildCommitId = Schema.Schema.Type<typeof LlamaBuildCommitId>
export const LlamaDeviceId = id("LlamaDeviceId", 512)
export type LlamaDeviceId = Schema.Schema.Type<typeof LlamaDeviceId>
export const LlamaInstanceId = id("LlamaInstanceId")
export type LlamaInstanceId = Schema.Schema.Type<typeof LlamaInstanceId>
export const LlamaServedModelId = id("LlamaServedModelId", 4096)
export type LlamaServedModelId = Schema.Schema.Type<typeof LlamaServedModelId>
export const LlamaModelRegistrationId = id("LlamaModelRegistrationId")
export type LlamaModelRegistrationId = Schema.Schema.Type<typeof LlamaModelRegistrationId>
export const LlamaExecutionProfileId = id("LlamaExecutionProfileId")
export type LlamaExecutionProfileId = Schema.Schema.Type<typeof LlamaExecutionProfileId>
export const LlamaFitPlanId = id("LlamaFitPlanId")
export type LlamaFitPlanId = Schema.Schema.Type<typeof LlamaFitPlanId>
export const LlamaOperationId = id("LlamaOperationId")
export type LlamaOperationId = Schema.Schema.Type<typeof LlamaOperationId>
export const ExternalServerConfigId = id("ExternalServerConfigId")
export type ExternalServerConfigId = Schema.Schema.Type<typeof ExternalServerConfigId>
