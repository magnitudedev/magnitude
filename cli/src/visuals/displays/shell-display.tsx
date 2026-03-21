import { TextAttributes } from '@opentui/core';
import { defineDisplay } from '@magnitudedev/tools';
import { shellModel, type ShellState } from '@magnitudedev/agent/src/models';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';
import { shortenCommandPreview } from '../../utils/strings';

const SHIMMER_INTERVAL_MS = 160;
const MAX_COMMAND_DISPLAY_LEN = 80;

export const shellDisplay = defineDisplay(shellModel, {
  render: ({ state, result, isExpanded, onToggle }) => {
    const theme = useTheme();
    const command = state.command || '';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const isRejected = state.phase === 'rejected';
    const isInterrupted = state.phase === 'interrupted';
    const isDetached = state.done === 'detached';

    const output = result?.status === 'success' && result.output
      ? [result.output.stdout, result.output.stderr].filter(Boolean).join('\n').trim()
      : result?.status === 'error'
        ? result.message ?? ''
        : '';

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text>
            <span style={{ fg: theme.muted }}>{'$ '}</span>
            <span style={{ fg: theme.foreground }}>
              {isExpanded ? command : shortenCommandPreview(command, MAX_COMMAND_DISPLAY_LEN)}
            </span>

            {isRunning ? (
              <>
                <span>{' '}</span>
                <ShimmerText text="running..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : isDetached ? (
              <span style={{ fg: theme.warning }}>{' · Detached'}</span>
            ) : isRejected ? (
              <span style={{ fg: theme.error }}>{' · Rejected'}</span>
            ) : isInterrupted ? (
              <span style={{ fg: theme.muted }}>{' · Interrupted'}</span>
            ) : isError ? (
              <span style={{ fg: theme.error }}>{' · Error'}</span>
            ) : state.phase === 'completed' ? (
              <span style={{ fg: theme.success }}>{' · Done'}</span>
            ) : null}

            {!isRunning && (
              <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                {isExpanded ? ' (collapse)' : ' (expand)'}
              </span>
            )}
          </text>
        </Button>

        {isExpanded && output.length > 0 && (
          <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
            {output}
          </text>
        )}


      </box>
    );
  },
  summary: (state) => {
    const command = state.command.trim();
    if (state.phase === 'streaming' || state.phase === 'executing') return command ? `$ ${command}` : 'Running shell command';
    if (state.phase === 'error') return command ? `Shell error: $ ${command}` : 'Shell error';
    if (state.phase === 'rejected') return command ? `Rejected: $ ${command}` : 'Shell rejected';
    if (state.done === 'detached') return 'Detached shell';
    return command ? `$ ${command}` : 'Ran shell command';
  },
});
