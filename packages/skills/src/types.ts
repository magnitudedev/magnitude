export interface WorkflowSkill {
  readonly name: string
  readonly description: string
  readonly preamble: string
  readonly phases: readonly Phase[]
}

export interface Phase {
  readonly name: string
  readonly prompt: string
  readonly submit?: SubmitBlock
  readonly criteria?: readonly Criteria[]
  readonly hooks?: Hooks
}

export interface SubmitBlock {
  readonly fields: readonly SubmitField[]
}

export type SubmitField =
  | { readonly type: 'text'; readonly name: string; readonly description: string }
  | { readonly type: 'file'; readonly name: string; readonly fileType?: string; readonly description: string }

export type Criteria =
  | { readonly type: 'shell-succeed'; readonly name: string; readonly command: string }
  | { readonly type: 'user-approval'; readonly name: string; readonly message: string }
  | { readonly type: 'agent-approval'; readonly name: string; readonly subagent: string; readonly prompt: string }

export type CriteriaResult =
  | { readonly type: 'passed' }
  | { readonly type: 'failed'; readonly reason: string }

export type FieldError =
  | { readonly type: 'missing'; readonly name: string }
  | { readonly type: 'file-not-found'; readonly name: string; readonly path: string }

export type ValidationResult =
  | { readonly valid: true }
  | { readonly valid: false; readonly errors: readonly FieldError[] }

export type WorkflowAction =
  | { readonly type: 'submit'; readonly fields: ReadonlyMap<string, string> }
  | { readonly type: 'advance' }
  | { readonly type: 'criteria-failed'; readonly results: readonly CriteriaResult[] }

export interface Hooks {
  readonly onStart?: string
  readonly onSubmit?: string
  readonly onAccept?: string
  readonly onReject?: string
}
