import { Git as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { query } from "./bind";

const GetGitRecentFiles = query(
  Rpcs.getGitRecentFiles,
  (client) => client.git.getGitRecentFiles
);

export const Git = Group.make({ GetGitRecentFiles });
