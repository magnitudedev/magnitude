import { TextAttributes } from '@opentui/core';
import { type FileTreeState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { useTheme } from '../../hooks/use-theme';

const SHIMMER_INTERVAL_MS = 160;

export const fileTreeDisplay = createToolDisplay<FileTreeState>('fileTree', {
  render: ({ state, isExpanded, onToggle }) => {
    const theme = useTheme();
    const isRunning = state.phase === 'streaming' || state.phase === 'executing';
    const isError = state.phase === 'error';
    const fileCount = state.fileCount;
    const dirCount = state.dirCount;

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button onClick={onToggle}>
          <text style={{ wrapMode: 'word' }}>
            <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '◫ '}</span>
            {isRunning ? (
              <>
                <span style={{ fg: theme.foreground }}>{'Listing '}</span>
                <span style={{ fg: theme.muted }}>{state.path || '...'}</span>
                <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
              </>
            ) : isError ? (
              <>
                <span style={{ fg: theme.foreground }}>{'List '}</span>
                <span style={{ fg: theme.muted }}>{state.path}</span>
                <span style={{ fg: theme.error }}>{' · Error'}</span>
                <span style={{ fg: theme.muted }}>{` (${state.errorDetail || ''})`}</span>
              </>
            ) : (
              <>
                <span style={{ fg: theme.foreground }}>{'Listed '}</span>
                <span style={{ fg: theme.muted }}>{state.path}</span>
                {state.entries.length > 0 ? (
                  <>
                    <span style={{ fg: theme.info }}>
                      {` · ${fileCount} ${fileCount === 1 ? 'file' : 'files'}`}
                      {dirCount > 0 ? `, ${dirCount} ${dirCount === 1 ? 'dir' : 'dirs'}` : ''}
                    </span>
                    <span style={{ fg: theme.secondary }} attributes={TextAttributes.DIM}>
                      {isExpanded ? ' (collapse)' : ' (expand)'}
                    </span>
                  </>
                ) : (
                  <span style={{ fg: theme.muted }}>{' · empty'}</span>
                )}
              </>
            )}
          </text>
        </Button>

        {isExpanded && state.entries.length > 0 && (
          <box style={{ flexDirection: 'column', paddingLeft: 2 }}>
            {state.entries.map((entry, i) => (
              <text key={i}>
                <span style={{ fg: theme.muted }}>{'  '.repeat(entry.depth)}</span>
                {entry.type === 'dir' ? (
                  <span style={{ fg: theme.directory }}>{entry.name}/</span>
                ) : (
                  <span style={{ fg: theme.muted }}>{entry.name}</span>
                )}
              </text>
            ))}
          </box>
        )}
      </box>
    );
  },
  summary: (state) => {
    const path = state.path || 'files';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Listing ${path}`;
    if (state.phase === 'completed') return `Listed ${path}`;
    return `List ${path}`;
  },
});
