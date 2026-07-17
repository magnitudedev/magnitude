import { Schema } from 'effect'
import { ContextImagePartSchema, ContextPartSchema } from './content'

export const AgentImageMediaTypeSchema = Schema.Literal(
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
)
export type AgentImageMediaType = typeof AgentImageMediaTypeSchema.Type

export const AgentImageAttachmentSchema = Schema.Struct({
  type: Schema.Literal('image'),
  image: ContextImagePartSchema,
})
export type AgentImageAttachment = typeof AgentImageAttachmentSchema.Type

export const AgentMentionFileAttachmentSchema = Schema.Struct({
  type: Schema.Literal('mention_file'),
  path: Schema.String,
})
export type AgentMentionFileAttachment = typeof AgentMentionFileAttachmentSchema.Type

export const AgentMentionFileRangeAttachmentSchema = Schema.Struct({
  type: Schema.Literal('mention_file_range'),
  path: Schema.String,
  startLine: Schema.Number,
  endLine: Schema.Number,
})
export type AgentMentionFileRangeAttachment = typeof AgentMentionFileRangeAttachmentSchema.Type

export const AgentMentionDirectoryAttachmentSchema = Schema.Struct({
  type: Schema.Literal('mention_directory'),
  path: Schema.String,
})
export type AgentMentionDirectoryAttachment = typeof AgentMentionDirectoryAttachmentSchema.Type

export const AgentMentionAttachmentSchema = Schema.Union(
  AgentMentionFileAttachmentSchema,
  AgentMentionFileRangeAttachmentSchema,
  AgentMentionDirectoryAttachmentSchema,
)
export type AgentMentionAttachment = typeof AgentMentionAttachmentSchema.Type

export const MentionPlacementSchema = Schema.Union(
  Schema.TaggedStruct('inline', {
    start: Schema.Number,
    end: Schema.Number,
  }),
  Schema.TaggedStruct('trailing', {}),
)
export type MentionPlacement = typeof MentionPlacementSchema.Type

export const MentionOccurrenceSchema = Schema.Struct({
  occurrenceId: Schema.String,
  attachment: AgentMentionAttachmentSchema,
  placement: MentionPlacementSchema,
})
export type MentionOccurrence = typeof MentionOccurrenceSchema.Type

export const MentionResolutionSchema = Schema.Union(
  Schema.Struct({
    occurrenceId: Schema.String,
    status: Schema.Literal('resolved'),
    parts: Schema.Array(ContextPartSchema),
    truncated: Schema.Boolean,
  }),
  Schema.Struct({
    occurrenceId: Schema.String,
    status: Schema.Literal('failed'),
    reason: Schema.String,
  }),
)
export type MentionResolution = typeof MentionResolutionSchema.Type
