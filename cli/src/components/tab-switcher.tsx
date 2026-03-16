import { memo, useState, useEffect, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import { Button } from './button'
import { useTheme } from '../hooks/use-theme'
import { GREEN_PULSE } from '../utils/agent-colors'
import { red } from '../utils/palette'

const PULSE_INTERVAL_MS = 200

interface TabSwitcherProps {
  activeTab: 'main' | 'agents'
  hasActiveAgents: boolean
  hasUnreadMain: boolean
  onSwitch: (tab: 'main' | 'agents') => void
}

export const TabSwitcher = memo(function TabSwitcher({
  activeTab,
  hasActiveAgents,
  hasUnreadMain,
  onSwitch,
}: TabSwitcherProps) {
  const theme = useTheme()
  const [pulseIndex, setPulseIndex] = useState(0)
  const pulseRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (hasActiveAgents) {
      pulseRef.current = setInterval(() => {
        setPulseIndex(prev => (prev + 1) % GREEN_PULSE.length)
      }, PULSE_INTERVAL_MS)
      return () => {
        if (pulseRef.current) clearInterval(pulseRef.current)
      }
    }
  }, [hasActiveAgents])

  const mainActive = activeTab === 'main'
  const agentsActive = activeTab === 'agents'
  const [mainHovered, setMainHovered] = useState(false)
  const [agentsHovered, setAgentsHovered] = useState(false)

  const mainColor = mainActive ? theme.foreground : (mainHovered ? theme.foreground : theme.muted)
  const agentsColor = agentsActive ? theme.foreground : (agentsHovered ? theme.foreground : theme.muted)

  return (
    <box style={{ flexDirection: 'row', alignItems: 'center', flexShrink: 0 }}>
      {/* Label */}
      <text style={{ fg: theme.muted, wrapMode: 'none' }}>{'View: '}</text>

      {/* Main tab */}
      <Button onClick={() => onSwitch('main')} onMouseOver={() => setMainHovered(true)} onMouseOut={() => setMainHovered(false)}>
        <text style={{ wrapMode: 'none' }}>
          <span fg={mainColor}>{'['}</span>
          <span
            fg={mainColor}
            attributes={mainActive ? TextAttributes.BOLD : 0}
          >
            {'Main'}
          </span>
          {hasUnreadMain && <span fg={red[400]}>{'●'}</span>}
          <span fg={mainColor}>{']'}</span>
        </text>
      </Button>

      {/* Separator */}
      <text style={{ fg: theme.muted, wrapMode: 'none' }}>{' │ '}</text>

      {/* Agents tab */}
      <Button onClick={() => onSwitch('agents')} onMouseOver={() => setAgentsHovered(true)} onMouseOut={() => setAgentsHovered(false)}>
        <text style={{ wrapMode: 'none' }}>
          <span fg={agentsColor}>{'['}</span>
          <span
            fg={agentsColor}
            attributes={agentsActive ? TextAttributes.BOLD : 0}
          >
            {'Agents'}
          </span>
          {hasActiveAgents && <span fg={GREEN_PULSE[pulseIndex]}>{'●'}</span>}
          <span fg={agentsColor}>{']'}</span>
        </text>
      </Button>
    </box>
  )
})