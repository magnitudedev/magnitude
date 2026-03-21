export interface EditDiff {
  startLine: number;
  removedLines: readonly string[];
  addedLines: readonly string[];
  contextBefore: readonly string[];
  contextAfter: readonly string[];
}
