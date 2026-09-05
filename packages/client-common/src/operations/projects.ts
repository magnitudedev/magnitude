import { Projects as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { mutation, query } from "./bind";

const ListProjects = query(
  Rpcs.listProjects,
  (client) => client.projects.listProjects,
  { staleTime: Infinity }
);

const CreateProject = mutation(
  Rpcs.createProject,
  (client) => client.projects.createProject
);

const EditProject = mutation(
  Rpcs.editProject,
  (client) => client.projects.editProject
);

const RemoveProject = mutation(
  Rpcs.removeProject,
  (client) => client.projects.removeProject
);

const RestoreProject = mutation(
  Rpcs.restoreProject,
  (client) => client.projects.restoreProject
);

const RevealProjectSource = mutation(
  Rpcs.revealProjectSource,
  (client) => client.projects.revealProjectSource
);

const InspectProject = query(
  Rpcs.inspectProject,
  (client) => client.projects.inspectProject,
  { staleTime: Infinity }
);

export const Projects = Group.make({
  ListProjects,
  CreateProject,
  EditProject,
  RemoveProject,
  RestoreProject,
  RevealProjectSource,
  InspectProject,
});
