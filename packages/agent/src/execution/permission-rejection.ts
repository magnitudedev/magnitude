/**
 * Permission Rejection Types
 *
 * Typed gate rejection reasons for the permission system.
 * js-act carries these through generically — Magnitude defines the vocabulary and policy.
 *
 * Cancellation policy (handled in execution-manager.ts GateRejected handler):
 * - ReadonlyMode → cancelled: false (agent adjusts, continues planning)
 * - Forbidden → cancelled: false (agent tries different approach, continues)
 * - DangerousInBuild → cancelled: false (agent tries safer alternative, continues autonomously)
 * - OutsideCwd → cancelled: false (agent stays in cwd, continues)
 * - UserRejection → cancelled: true (agent stops turn, yields to user)
 */

import { Data } from 'effect'

export type PermissionRejection = Data.TaggedEnum<{
  /** Plan mode — file modifications not allowed until requestBuild() is called */
  readonly ReadonlyMode: { readonly reason: string }
  /** Command is forbidden in all modes */
  readonly Forbidden: { readonly reason: string }
  /** Dangerous command auto-rejected in build mode (not inherently wrong, just can't run autonomously) */
  readonly DangerousInBuild: { readonly reason: string }
  /** Operation targets path outside working directory */
  readonly OutsideCwd: { readonly reason: string }
  /** User explicitly rejected approval via UI */
  readonly UserRejection: { readonly reason: string }
}>

const permissionRejectionEnum = Data.taggedEnum<PermissionRejection>()

type PermissionRejectionReason = { readonly reason: string }

export const PermissionRejection = {
  ReadonlyMode: (
    args: PermissionRejectionReason,
  ): Extract<PermissionRejection, { readonly _tag: "ReadonlyMode" }> => ({
    _tag: "ReadonlyMode",
    ...args,
  }),
  Forbidden: (
    args: PermissionRejectionReason,
  ): Extract<PermissionRejection, { readonly _tag: "Forbidden" }> => ({
    _tag: "Forbidden",
    ...args,
  }),
  DangerousInBuild: (
    args: PermissionRejectionReason,
  ): Extract<PermissionRejection, { readonly _tag: "DangerousInBuild" }> => ({
    _tag: "DangerousInBuild",
    ...args,
  }),
  OutsideCwd: (
    args: PermissionRejectionReason,
  ): Extract<PermissionRejection, { readonly _tag: "OutsideCwd" }> => ({
    _tag: "OutsideCwd",
    ...args,
  }),
  UserRejection: (
    args: PermissionRejectionReason,
  ): Extract<PermissionRejection, { readonly _tag: "UserRejection" }> => ({
    _tag: "UserRejection",
    ...args,
  }),
  $is: permissionRejectionEnum.$is,
  $match: permissionRejectionEnum.$match,
}
