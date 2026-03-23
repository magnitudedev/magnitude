import { TextAttributes } from '@opentui/core';
import { type ShellState } from '@magnitudedev/agent/src/models';
import type { CommonToolProps } from '../types';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';
import { shortenCommandPreview } from '../../utils/strings';

const SHIMMER_INTERVAL_MS = 160;
const RESULT_TRUNCATE_LEN = 80;
const MAX_COMMAND_DISPLAY_LEN = 80;

function truncateLine(text: string, max: number): string {
  if (!text) return '';
  const firstLine = text.split('\n').find(l => l.trim() !== '') ?? '';
  if (firstLine.length > max) return firstLine.slice(0, max - 3) + '...';
  return firstLine;
}

export const shellDisplay = {
  render({ state, isExpanded, onToggle }: { state: ShellState } & CommonToolProps) {
    const theme = useTheme();
    const command = state.command || '';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isRejected = state.phase === 'rejected';
    const isInterrupted = state.phase === 'interrupted';
    const isDetached = state.done === 'detached';
    const isCompleted = state.phase === 'completed';
    const isError = state.phase === 'error';
    const isSuccess = isCompleted || isDetached;

    const isFailed = isError || (isSuccess && state.exitCode !== undefined && state.exitCode !== 0);
    const rejectionReason = isRejected ? state.rejectionReason : undefined;
    const detachedPid = isDetached ? state.pid : undefined;

    const resultPreview = isSuccess
      ? (isFailed ? (state.stderr || state.stdout) : (state.stdout || state.stderr))
      : isError
        ? state.errorMessage ?? ''
        : '';

    const fullResultText = isSuccess
      ? [state.stdout, isFailed ? state.stderr : ''].filter(Boolean).join('\n').replace(/^\n+/, '').trimEnd()
      : isError
        ? state.errorMessage ?? ''
        : '';

    return (
      <box style={{ flexDirection: 'column' }}>
        {isInterrupted ? (
          <text>
            <span style={{ fg: theme.muted }}>{'$ '}</span>
            <span style={{ fg: theme.foreground }}>
              {shortenCommandPreview(command, MAX_COMMAND_DISPLAY_LEN)}
            </span>
            <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{' · Interrupted'}</span>
          </text>
        ) : (
          <Button onClick={onToggle}>
            <text>
              <span style={{ fg: theme.muted }}>{'$ '}</span>
              <span style={{ fg: theme.foreground }}>
                {isExpanded ? command : shortenCommandPreview(command, MAX_COMMAND_DISPLAY_LEN)}
              </span>
              {isRunning ? (
                <>
                  {'  '}
                  <ShimmerText text="running..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
                </>
              ) : isRejected ? (
                rejectionReason
                  ? <><span style={{ fg: theme.error }}>{' · System Rejected'}</span><span style={{ fg: theme.muted }}>{` (${rejectionReason})`}</span></>
                  : <span style={{ fg: theme.error }}>{' · User Rejected'}</span>
              ) : isDetached ? (
                <>
                  <span style={{ fg: theme.warning }}>{` · ⇀ Detached (PID ${detachedPid ?? '?'})`}</span>
                  <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                    {isExpanded ? ' · (collapse)' : ' · (expand)'}
                  </span>
                </>
              ) : (isCompleted || isError) ? (
                <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                  {isExpanded ? ' · (collapse)' : ' · (expand)'}
                </span>
              ) : null}
            </text>
          </Button>
        )}

        {isFailed && !isExpanded && resultPreview && String(resultPreview).trim() && (
          <text style={{ fg: theme.error }} attributes={TextAttributes.DIM}>
            {'✗ '}{truncateLine(String(resultPreview), RESULT_TRUNCATE_LEN)}
          </text>
        )}

        {isExpanded && (isCompleted || isDetached || isError) && fullResultText.trim() && (
          <text style={{ fg: isDetached ? theme.warning : isFailed ? theme.error : theme.muted }} attributes={TextAttributes.DIM}>
            {isDetached ? '' : isFailed ? '✗ ' : ''}{fullResultText}
          </text>
        )}
      </box>
    );
  },
  summary(state: ShellState): string {
    const command = state.command.trim();
    if (state.phase === 'streaming' || state.phase === 'executing') return command ? `$ ${command}` : 'Running shell command';
    if (state.phase === 'error') return command ? `Shell error: $ ${command}` : 'Shell error';
    if (state.phase === 'rejected') return command ? `Rejected: $ ${command}` : 'Shell command rejected';
    if (state.done === 'detached') return 'Detached shell';
    return command ? `$ ${command}` : 'Ran shell command';
  },
};
