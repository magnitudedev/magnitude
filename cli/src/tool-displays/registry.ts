import type { Display } from './types'
import { getDisplayMap } from './types'

// Import all display files to trigger self-registration via createToolDisplay
import './displays/default'
import './displays/shell'
import './displays/file-read'
import './displays/file-tree'
import './displays/file-search'
import './displays/diff'
import './displays/content'
import './displays/web-search'
import './displays/web-fetch'
import './displays/browser-action'
import './displays/agent-create'
import './displays/agent-kill'
import './displays/skill'
import './displays/phase-submit'
import './displays/phase-verdict'

export const displayRegistry = {
  get(toolKey: string): Display<any, any> {
    const map = getDisplayMap()
    return map[toolKey] ?? map['default']
  },
}
