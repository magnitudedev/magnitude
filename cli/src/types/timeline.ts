import type { DisplayMessage } from '@magnitudedev/agent'
import type { BashResult } from '../utils/bash-executor'

export interface ChatMessageItem {
  readonly kind: 'chat'
  readonly id: string
  readonly timestamp: number
  readonly message: DisplayMessage
}

export interface BashOutputItem {
  readonly kind: 'bash'
  readonly id: string
  readonly timestamp: number
  readonly result: BashResult
}

export interface SystemMessageItem {
  readonly kind: 'system'
  readonly id: string
  readonly text: string
  readonly timestamp: number
}

export type TimelineItem = ChatMessageItem | BashOutputItem | SystemMessageItem
