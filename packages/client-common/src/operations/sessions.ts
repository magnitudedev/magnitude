import { Sessions as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { type ActiveSessionStatuses as ActiveSessionStatusesSnapshot } from "@magnitudedev/sdk";
import { turnAdmissionScope } from "./configuration";
import { mutation, query, streamQuery } from "./bind";

const ListSessions = query(
  Rpcs.listSessions,
  (client) => client.sessions.listSessions,
  { staleTime: Infinity }
);

const ListRecentSessionDirectories = query(
  Rpcs.listRecentSessionDirectories,
  (client) => client.sessions.listRecentSessionDirectories,
  { staleTime: Infinity }
);

const StreamActiveSessionStatuses = streamQuery(
  Rpcs.streamActiveSessionStatuses,
  client => client.sessions.streamActiveSessionStatuses,
  {
    reduce: (_, snapshot): ActiveSessionStatusesSnapshot => snapshot,
  }
);

const CreateSession = mutation(
  Rpcs.createSession,
  (client) => client.sessions.createSession,
  { scope: () => turnAdmissionScope }
);

const PreloadSession = mutation(
  Rpcs.preloadSession,
  (client) => client.sessions.preloadSession
);

const ReleaseSessionPreload = mutation(
  Rpcs.releaseSessionPreload,
  (client) => client.sessions.releaseSessionPreload
);

const GetSession = query(
  Rpcs.getSession,
  (client) => client.sessions.getSession,
  { staleTime: Infinity }
);

const DeleteArchivedSession = mutation(
  Rpcs.deleteArchivedSession,
  (client) => client.sessions.deleteArchivedSession
);

const ArchiveSession = mutation(
  Rpcs.archiveSession,
  (client) => client.sessions.archiveSession
);

const RestoreSession = mutation(
  Rpcs.restoreSession,
  (client) => client.sessions.restoreSession
);

const SetSessionPinned = mutation(
  Rpcs.setSessionPinned,
  (client) => client.sessions.setSessionPinned
);

export const Sessions = Group.make({
  ListSessions,
  ListRecentSessionDirectories,
  StreamActiveSessionStatuses,
  CreateSession,
  PreloadSession,
  ReleaseSessionPreload,
  GetSession,
  DeleteArchivedSession,
  ArchiveSession,
  RestoreSession,
  SetSessionPinned,
});
