import { useMemo } from 'react';
import { TextAttributes } from '@opentui/core';
import { defineDisplay } from '@magnitudedev/tools';
import { diffModel, type DiffState } from '@magnitudedev/agent/src/models';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { DiffHunk } from '../../components/diff-hunk';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const diffDisplay = defineDisplay(diffModel, {
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const path = state.path;
    const oldText = state.oldText ?? '';
    const newText = state.newText ?? '';
    const oldLines = oldText.length > 0 ? oldText.split('\n') : [];
    const newLines = newText.length > 0 ? newText.split('\n') : [];

    const totals = useMemo(() => {
      const added = state.diffs.reduce((sum, d) => sum + d.addedLines.length, 0);
      const removed = state.diffs.reduce((sum, d) => sum + d.removedLines.length, 0);
      return { added, removed };
    }, [state.diffs]);

    const isStreaming = state.phase === 'streaming' || state.phase === 'executing';
    const isDone = state.phase === 'completed';
    const isError = state.phase === 'error' || state.phase === 'rejected' || state.phase === 'interrupted';

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '✎ '}</span>
            {isDone ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Edited '}</span>
                <span style={{ fg: theme.primary }} attributes={TextAttributes.UNDERLINE}>{String(path ?? 'file')}</span>
              </>
            ) : (
              <>
                <span style={{ fg: theme.foreground }}>{'Editing '}</span>
                <span style={{ fg: theme.muted }}>{String(path ?? '...')}</span>
                {!isError && <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />}
              </>
            )}
          </text>
        </Button>

        {isDone && state.diffs.length > 0 && (
          <Button onClick={onToggle}>
            <text>
              <span style={{ fg: theme.syntax.string }} attributes={TextAttributes.DIM}>{` +${totals.added}`}</span>
              <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>{'/'}</span>
              <span style={{ fg: theme.error }} attributes={TextAttributes.DIM}>{`-${totals.removed}`}</span>
              <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                {isExpanded ? ' (collapse)' : ' (expand)'}
              </span>
            </text>
          </Button>
        )}

        {isStreaming && oldLines.length > 0 && (
          <DiffHunk
            removedLines={oldLines}
            addedLines={newLines}
            streamingCursor
          />
        )}

        {isDone && isExpanded && state.diffs.length > 0 && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {state.diffs.map((diff, index) => (
              <DiffHunk
                key={`${String(path ?? 'file')}-${index}`}
                contextBefore={[...diff.contextBefore]}
                removedLines={[...diff.removedLines]}
                addedLines={[...diff.addedLines]}
                contextAfter={[...diff.contextAfter]}
              />
            ))}
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const path = state.path ?? 'file';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Editing ${String(path)}`;
    if (state.phase === 'completed') return `Edited ${String(path)}`;
    if (state.phase === 'error') return `Edit error ${String(path)}`;
    if (state.phase === 'rejected') return `Edit rejected ${String(path)}`;
    return `Edit interrupted ${String(path)}`;
  },
});
