import { useState } from 'react';
import { TextAttributes } from '@opentui/core';
import { type SkillActivationState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const skillDisplay = createToolDisplay<SkillActivationState>({
  render: ({ state, onFileClick }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const [isHovered, setIsHovered] = useState(false);

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button
          onClick={() => { if (state.skillPath) onFileClick?.(state.skillPath) }}
          onMouseOver={() => setIsHovered(true)}
          onMouseOut={() => setIsHovered(false)}
        >
          <box style={{ flexDirection: 'column' }}>
            <text style={{ wrapMode: 'word' }}>
              <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '[▸] '}</span>
              {isRunning ? (
                <>
                  <span style={{ fg: theme.foreground }}>{'Activating skill '}</span>
                  <span style={{ fg: isHovered && state.skillPath ? theme.link : theme.primary }} attributes={state.skillPath ? TextAttributes.UNDERLINE : undefined}>{state.skillName || '...'}</span>
                  <ShimmerText text=" ..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.info} />
                </>
              ) : isError ? (
                <>
                  <span style={{ fg: theme.foreground }}>{'Activated skill '}</span>
                  <span style={{ fg: theme.muted }}>{state.skillName || 'skill'}</span>
                  <span style={{ fg: theme.error }}>{' · Error'}</span>
                  {state.errorDetail ? <span style={{ fg: theme.muted }}>{` (${state.errorDetail})`}</span> : null}
                </>
              ) : (
                <>
                  <span style={{ fg: theme.foreground }}>{'Skill activated '}</span>
                  <span style={{ fg: isHovered && state.skillPath ? theme.link : theme.primary }} attributes={state.skillPath ? TextAttributes.UNDERLINE : undefined}>{state.skillName || 'skill'}</span>
                </>
              )}
            </text>
          </box>
        </Button>
      </box>
    );
  },
  summary: (state) => {
    const target = state.skillName ?? 'skill';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Activating skill ${target}`;
    return `Skill activated ${target}`;
  },
});
