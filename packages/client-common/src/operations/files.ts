import { Files as Rpcs } from "@magnitudedev/sdk";
import { Group } from "@magnitudedev/effect-query";
import { mutation, query, subscription } from "./bind";

const UploadAttachment = mutation(
  Rpcs.uploadAttachment,
  (client) => client.files.uploadAttachment
);

const ListFiles = query(Rpcs.listFiles, (client) => client.files.listFiles);

const ReadFile = query(Rpcs.readFile, (client) => client.files.readFile);

const CheckFileExists = query(
  Rpcs.checkFileExists,
  (client) => client.files.checkFileExists
);

const WatchFile = subscription(
  Rpcs.watchFile,
  (client) => client.files.watchFile,
  { gcTime: "30 seconds" }
);

const ResolvePath = query(
  Rpcs.resolvePath,
  (client) => client.files.resolvePath
);

const SearchMentions = query(
  Rpcs.searchMentions,
  (client) => client.files.searchMentions,
  { gcTime: "30 seconds" }
);

const SearchDirectories = query(
  Rpcs.searchDirectories,
  (client) => client.files.searchDirectories,
  { gcTime: "30 seconds" }
);

export const Files = Group.make({
  UploadAttachment,
  ListFiles,
  ReadFile,
  CheckFileExists,
  WatchFile,
  ResolvePath,
  SearchMentions,
  SearchDirectories,
});
