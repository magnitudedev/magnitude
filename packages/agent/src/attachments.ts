import { Schema } from 'effect'

export const AgentImageMediaTypeSchema = Schema.Literal(
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
)
export type AgentImageMediaType = typeof AgentImageMediaTypeSchema.Type

export const AgentImageAttachmentSchema = Schema.Struct({
  type: Schema.Literal('image'),
  path: Schema.String,
  filename: Schema.String,
  mediaType: AgentImageMediaTypeSchema,
  width: Schema.Number,
  height: Schema.Number,
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

export const AgentMessageAttachmentSchema = Schema.Union(
  AgentImageAttachmentSchema,
  AgentMentionAttachmentSchema,
)
export type AgentMessageAttachment = typeof AgentMessageAttachmentSchema.Type

export const MentionResolutionSchema = Schema.Union(
  Schema.Struct({
    status: Schema.Literal('resolved'),
    attachment: AgentMentionAttachmentSchema,
    content: Schema.String,
    truncated: Schema.Boolean,
    originalBytes: Schema.Number,
  }),
  Schema.Struct({
    status: Schema.Literal('failed'),
    attachment: AgentMentionAttachmentSchema,
    reason: Schema.String,
  }),
)
export type MentionResolution = typeof MentionResolutionSchema.Type
