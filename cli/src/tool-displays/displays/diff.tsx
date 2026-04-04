import { useMemo, useState } from 'react';
import { TextAttributes } from '@opentui/core';
import { type DiffState } from '@magnitudedev/agent/src/models';
import { createToolDisplay } from '../types';
import { Button } from '../../components/button';
import { ShimmerText } from '../../components/shimmer-text';
import { DiffHunk } from '../../components/diff-hunk';
import { useStreamingReveal } from '../../hooks/use-streaming-reveal';
import { useTheme } from '../../hooks/use-theme';
import { useSelectedFile } from '../../hooks/use-file-viewer';


const SHIMMER_INTERVAL_MS = 160;

export const diffDisplay = createToolDisplay<DiffState>({
  render: ({ state, isExpanded, onToggle, onFileClick }) => {
    const theme = useTheme();
    const path = state.path;
    const newText = state.newText ?? '';
    const [isHovered, setIsHovered] = useState(false);
    const [isExpandHovered, setIsExpandHovered] = useState(false);
    const selectedFile = useSelectedFile();
    const isViewerShowingSameFile = !!path && selectedFile?.path === path;

    const totals = useMemo(() => {
      const added = state.diffs.reduce((sum, d) => sum + d.addedLines.length, 0);
      const removed = state.diffs.reduce((sum, d) => sum + d.removedLines.length, 0);
      return { added, removed };
    }, [state.diffs]);

    const isStreaming = state.phase === 'streaming' || state.phase === 'executing';
    const isDone = state.phase === 'completed';
    const isError = state.phase === 'error' || state.phase === 'rejected' || state.phase === 'interrupted';

    const isStreamingNew = isStreaming && state.streamingTarget === 'new';

    // Progressive reveal of new text
    const { displayedContent: revealedNewText } = useStreamingReveal(newText, isStreamingNew);

    const streamingDiff = state.diffs[0];
    const showStreamingDiff = isStreaming && !!streamingDiff;

    if (isDone) {
      return (
        <box style={{ flexDirection: 'column' }}>
          <box style={{ flexDirection: 'row' }}>
            <text>
              <span style={{ fg: theme.info }}>{'✎ '}</span>
              <span style={{ fg: theme.foreground }}>{'Edited '}</span>
            </text>
            <Button
              onClick={() => { if (path) onFileClick?.(path) }}
              onMouseOver={() => setIsHovered(true)}
              onMouseOut={() => setIsHovered(false)}
            >
              <text>
                <span style={{ fg: isHovered ? (theme.link) : theme.primary }} attributes={TextAttributes.UNDERLINE}>{String(path ?? 'file')}</span>
              </text>
            </Button>
            {state.diffs.length > 0 && (
              <Button
                onClick={onToggle}
                onMouseOver={() => setIsExpandHovered(true)}
                onMouseOut={() => setIsExpandHovered(false)}
              >
                <text>
                  <span style={{ fg: theme.syntax.string }} attributes={isExpandHovered ? undefined : TextAttributes.DIM}>{` +${totals.added}`}</span>
                  <span style={{ fg: isExpandHovered ? theme.foreground : theme.secondary }} attributes={TextAttributes.DIM}>{'/'}</span>
                  <span style={{ fg: theme.error }} attributes={isExpandHovered ? undefined : TextAttributes.DIM}>{`-${totals.removed}`}</span>
                  <span style={{ fg: isExpandHovered ? theme.foreground : theme.secondary }} attributes={TextAttributes.DIM}>
                    {isExpanded ? ' (collapse)' : ' (expand)'}
                  </span>
                </text>
              </Button>
            )}
          </box>

          {isExpanded && state.diffs.length > 0 && (
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
    }

    return (
      <box style={{ flexDirection: 'column' }}>
        <Button
          onClick={() => { if (path) onFileClick?.(path) }}
          onMouseOver={() => setIsHovered(true)}
          onMouseOut={() => setIsHovered(false)}
        >
          <box style={{ flexDirection: 'column' }}>
            <text style={{ wrapMode: 'word' }}>
              <span style={{ fg: isError ? theme.error : theme.info }}>{isError ? '✗ ' : '✎ '}</span>
              {isError ? (
                <>
                  <span style={{ fg: theme.foreground }}>{'Edit '}</span>
                  <span style={{ fg: theme.muted }}>{String(path ?? '...')}</span>
                  <span style={{ fg: theme.error }}>{' · Error'}</span>
                </>
              ) : (
                <>
                  <span style={{ fg: theme.foreground }}>{'Editing '}</span>
                  <span style={{ fg: theme.muted }}>{String(path ?? '...')}</span>
                  <ShimmerText text="..." interval={SHIMMER_INTERVAL_MS} primaryColor={theme.secondary} />
                </>
              )}
            </text>
            {showStreamingDiff && !isViewerShowingSameFile && (
              <DiffHunk
                contextBefore={streamingDiff.contextBefore}
                removedLines={streamingDiff.removedLines}
                addedLines={isStreamingNew ? revealedNewText.split('\n') : streamingDiff.addedLines}
                contextAfter={streamingDiff.contextAfter}
                streamingCursor={isStreamingNew}
              />
            )}
          </box>
        </Button>
      </box>
    );
  },
  summary: (state) => {
    const path = state.path || 'file';
    if (state.phase === 'streaming' || state.phase === 'executing') return `Editing ${String(path)}`;
    return `Edited ${String(path)}`;
  },
});
