// Barrel export for @magnitudedev/client-common

// State
export * from './state/stream-errors'
export * from './state/agent-client'
export * from './state/agent-client-context'
export * from './state/session-atoms'
export * from './state/display-state-store'
export * from './state/file-watch-atom'
export * from './state/active-session-statuses'
export * from './display-view-controller/hooks'

// Platform
export * from './platform/types'
export * from './platform/platform-context'

// Stores
export * from './stores/tick-store'
export * from './stores/animation-tick-store'
export * from './stores/system-message-store'
export * from './stores/ephemeral-message-store'

// Sync layer
export * from './sync/index'

// Types
export * from './types/store'
export type { KeyEvent } from './types/key-event'
export * from './types/menu-action'

// Utils
export * from './utils/format-tokens'
export * from './utils/format-elapsed'
export * from './utils/palette'
export * from './utils/diff-utils'
export * from './utils/color-conversion'
export * from './utils/ascii-logo'
export * from './utils/task-tree'
export * from './utils/task-visual-status'
export * from './utils/attachment-overflow'
export * from './utils/strings'
export * from './utils/bash-executor'
export * from './utils/presentation-helpers'
export * from './utils/tool-summary-label'
export * from './utils/file-panel-utils'
export * from './utils/model-properties'
export * from './utils/model-slots'
export * from './utils/hardware-memory'
export * from './attachments/images'

// Commands
export * from './commands/slash-commands'
export * from './commands/command-router'

// Data
export * from './data/recent-chats'

// Markdown
export * from './markdown/parse'
export * from './markdown/theme'

// Chat
export * from './chat/submit-routing'
export * from './chat/paste/types'
export * from './chat/paste/apply'
export * from './chat/paste/effects'
export * from './chat/paste/content-resolver'
export * from './chat/paste/ingest-coordinator'
export * from './chat/task-list/types'

// Hooks
export * from './hooks/use-file-mentions'
export * from './hooks/use-slash-commands'
export * from './hooks/use-recent-chats-navigation'
export * from './hooks/use-copy-feedback'
export * from './hooks/use-esc-interrupt'
export * from './hooks/use-interrupt-actions'
export * from './hooks/use-settings-state'
export * from './hooks/use-usage-state'
export * from './hooks/use-sessions-list'
export * from './hooks/use-paginated-sessions'
export * from './hooks/use-infinite-scroll'
export * from './hooks/use-session-actions'
export * from './hooks/use-session-preload'
export * from './hooks/draft-session-owner'
export * from './hooks/use-slot-profiles'
export * from './hooks/use-model-config'
export * from './hooks/use-composer-state'
export * from './hooks/use-file-panel-state'
export * from './hooks/use-menu-actions-core'
export * from './hooks/use-local-inference-state'
export * from './hooks/use-mirrored-state'
export * from './hooks/use-onboarding-state'
export * from './display-view-controller/timeline-scroll-controller'
