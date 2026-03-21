import { TextAttributes } from '@opentui/core';
import { defineDisplay } from '@magnitudedev/tools';
import { fileTreeModel, type FileTreeState } from '@magnitudedev/agent/src/models';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const fileTreeDisplay = defineDisplay(fileTreeModel, {
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const path = state.path ?? '.';
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '◫ '}</span>
            {isRunning ? (
              <>
                <span>{`Listing ${path}`}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : state.phase === 'completed' ? (
              <>
                <span>{`Listed ${path} · ${state.fileCount} files, ${state.dirCount} dirs`}</span>
                <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                  {isExpanded ? ' (collapse)' : ' (expand)'}
                </span>
              </>
            ) : state.phase === 'rejected' ? (
              <span>{`List ${path} · Rejected`}</span>
            ) : state.phase === 'interrupted' ? (
              <span>{`List ${path} · Interrupted`}</span>
            ) : (
              <span style={{ fg: theme.error }}>{`List ${path} · Error`}</span>
            )}
          </text>
        </Button>

        {state.phase === 'completed' && isExpanded && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {state.entries.map((entry, idx) => (
              <text key={`${entry.name}-${idx}`} style={{ wrapMode: 'word' }}>
                <span>{`${'  '.repeat(entry.depth)}`}</span>
                {entry.type === 'dir' ? (
                  <span style={{ fg: theme.primary }}>{`${entry.name}/`}</span>
                ) : (
                  <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{entry.name}</span>
                )}
              </text>
            ))}
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const path = state.path ?? '.';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Listing ${path}`;
    if (state.phase === 'completed') return `Listed ${path}`;
    if (state.phase === 'error') return `List ${path} error`;
    if (state.phase === 'rejected') return `List ${path} rejected`;
    return `List ${path} interrupted`;
  },
});
