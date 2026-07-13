import { readdir, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { buildExtractionTranscript } from "../transcript";
import type { AppEvent } from "../../events";

type SessionStats = {
  sessionId: string;
  eventCount: number;
  transcriptChars: number;
  transcriptLines: number;
};

function countLines(text: string): number {
  if (text.length === 0) return 0;
  return text.split("\n").length;
}

function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

function pad(value: string, width: number): string {
  if (value.length >= width) return value;
  return value + " ".repeat(width - value.length);
}

function printTable(rows: SessionStats[]) {
  const headers = ["Session ID", "Events", "Chars", "Lines"];
  const sessionWidth = Math.max(
    headers[0].length,
    ...rows.map((r) => r.sessionId.length),
  );
  const eventsWidth = Math.max(
    headers[1].length,
    ...rows.map((r) => formatNumber(r.eventCount).length),
  );
  const charsWidth = Math.max(
    headers[2].length,
    ...rows.map((r) => formatNumber(r.transcriptChars).length),
  );
  const linesWidth = Math.max(
    headers[3].length,
    ...rows.map((r) => formatNumber(r.transcriptLines).length),
  );

  const divider = `+-${"-".repeat(sessionWidth)}-+-${"-".repeat(eventsWidth)}-+-${"-".repeat(charsWidth)}-+-${"-".repeat(linesWidth)}-+`;

  console.log(divider);
  console.log(
    `| ${pad(headers[0], sessionWidth)} | ${pad(headers[1], eventsWidth)} | ${pad(headers[2], charsWidth)} | ${pad(headers[3], linesWidth)} |`,
  );
  console.log(divider);

  for (const row of rows) {
    console.log(
      `| ${pad(row.sessionId, sessionWidth)} | ${pad(formatNumber(row.eventCount), eventsWidth)} | ${pad(formatNumber(row.transcriptChars), charsWidth)} | ${pad(formatNumber(row.transcriptLines), linesWidth)} |`,
    );
  }

  console.log(divider);
}

async function loadSessionStats(sessionId: string, sessionsDir: string): Promise<SessionStats | null> {
  const eventsPath = join(sessionsDir, sessionId, "events.jsonl");

  try {
    const eventsJsonl = await readFile(eventsPath, "utf8");
    const events: AppEvent[] = eventsJsonl
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as AppEvent);

    const transcript = buildExtractionTranscript(events);

    return {
      sessionId,
      eventCount: events.length,
      transcriptChars: transcript.length,
      transcriptLines: countLines(transcript),
    };
  } catch (error) {
    console.warn(`Skipping ${sessionId}: could not read/parse events.jsonl`);
    console.warn(error);
    return null;
  }
}

async function main() {
  const sessionsDir = join(homedir(), ".magnitude", "sessions");
  const dirEntries = await readdir(sessionsDir, { withFileTypes: true });
  const sessionIds = dirEntries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  if (sessionIds.length === 0) {
    console.log(`No sessions found in ${sessionsDir}`);
    return;
  }

  const stats: SessionStats[] = [];
  for (const sessionId of sessionIds) {
    const result = await loadSessionStats(sessionId, sessionsDir);
    if (result) stats.push(result);
  }

  stats.sort((a, b) => b.transcriptChars - a.transcriptChars);

  printTable(stats);

  const totals = stats.reduce(
    (acc, row) => {
      acc.sessions += 1;
      acc.events += row.eventCount;
      acc.chars += row.transcriptChars;
      acc.lines += row.transcriptLines;
      return acc;
    },
    { sessions: 0, events: 0, chars: 0, lines: 0 },
  );

  console.log("");
  console.log("Summary");
  console.log(`- Sessions: ${formatNumber(totals.sessions)}`);
  console.log(`- Total events: ${formatNumber(totals.events)}`);
  console.log(`- Total transcript chars: ${formatNumber(totals.chars)}`);
  console.log(`- Total transcript lines: ${formatNumber(totals.lines)}`);

  if (totals.sessions > 0) {
    console.log(`- Avg chars/session: ${formatNumber(Math.round(totals.chars / totals.sessions))}`);
    console.log(`- Avg lines/session: ${formatNumber(Math.round(totals.lines / totals.sessions))}`);
  }
}

main().catch((error) => {
  console.error("Failed to compute transcript sizes:");
  console.error(error);
  process.exitCode = 1;
});