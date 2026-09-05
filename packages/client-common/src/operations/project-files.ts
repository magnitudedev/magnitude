import { ProjectFiles as Rpcs } from "@magnitudedev/sdk";
import { Effect } from "effect";
import { Group, QueryClient } from "@magnitudedev/effect-query";
import {
  parentDirectory,
  type ProjectId,
  type RelativePath,
} from "@magnitudedev/sdk";
import { mutation, query, subscription } from "./bind";

const ListProjectDirectory = query(
  Rpcs.listProjectDirectory,
  (client) => client.projectFiles.listProjectDirectory,
  { gcTime: "2 minutes" }
);

const WatchProjectFiles = subscription(
  Rpcs.watchProjectFiles,
  (client) => client.projectFiles.watchProjectFiles,
  { gcTime: "2 minutes" }
);

const ReadProjectFile = query(
  Rpcs.readProjectFile,
  (client) => client.projectFiles.readProjectFile
);

/** A committed write or deletion changes the exact file and its parent listing. */
const synchronizeProjectPath = (projectId: ProjectId, path: RelativePath) =>
  QueryClient.invalidate(ReadProjectFile.match({ projectId, path })).pipe(
    Effect.zipRight(
      QueryClient.invalidate(
        ListProjectDirectory.match({
          projectId,
          directory: parentDirectory(path),
        })
      )
    )
  );

const WriteProjectFile = mutation(
  Rpcs.writeProjectFile,
  (client) => client.projectFiles.writeProjectFile,
  {
    synchronize: (_, { projectId, path }) =>
      synchronizeProjectPath(projectId, path),
  }
);

const DeleteProjectFile = mutation(
  Rpcs.deleteProjectFile,
  (client) => client.projectFiles.deleteProjectFile,
  {
    synchronize: (_, { projectId, path }) =>
      synchronizeProjectPath(projectId, path),
  }
);

const MoveProjectEntry = mutation(
  Rpcs.moveProjectEntry,
  (client) => client.projectFiles.moveProjectEntry,
  {
    synchronize: () =>
      QueryClient.invalidate(ListProjectDirectory.match()).pipe(
        Effect.zipRight(QueryClient.invalidate(ReadProjectFile.match()))
      ),
  }
);

export const ProjectFiles = Group.make({
  ListProjectDirectory,
  WatchProjectFiles,
  ReadProjectFile,
  WriteProjectFile,
  DeleteProjectFile,
  MoveProjectEntry,
});
