import type { Display } from './display-types'
import { getDisplayMap } from './display-types'

// Import all display files to trigger self-registration via createToolDisplay
import './displays/default-display'
import './displays/shell-display'
import './displays/file-read-display'
import './displays/file-tree-display'
import './displays/file-search-display'
import './displays/diff-display'
import './displays/content-display'
import './displays/web-search-display'
import './displays/web-fetch-display'
import './displays/browser-action-display'
import './displays/agent-create-display'
import './displays/agent-kill-display'
import './displays/skill-display'

export const displayRegistry = {
  get(toolKey: string): Display<any, any> {
    const map = getDisplayMap()
    return map[toolKey] ?? map['default']
  },
}
