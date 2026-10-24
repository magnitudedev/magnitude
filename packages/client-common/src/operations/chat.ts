import {
  Chat as Rpcs,
  applyChatUpdate,
  type ChatConversation,
} from "@magnitudedev/sdk"
import { Option } from "effect"
import { Group } from "@magnitudedev/effect-query"
import { mutation, query, streamQuery } from "./bind"

const ListModels = query(Rpcs.listModels, (client) => client.chat.listModels, {
  staleTime: "10 seconds",
})
const ListConversations = query(
  Rpcs.listConversations,
  (client) => client.chat.listConversations
)
const WatchConversation = streamQuery(
  Rpcs.watchConversation,
  (client) => client.chat.watchConversation,
  {
    reduce: (previous, update): ChatConversation | null =>
      applyChatUpdate(Option.getOrNull(previous), update),
    gcTime: "5 seconds",
  }
)
const CreateConversation = mutation(
  Rpcs.createConversation,
  (client) => client.chat.createConversation
)
const UpdateConversation = mutation(
  Rpcs.updateConversation,
  (client) => client.chat.updateConversation
)
const DeleteConversation = mutation(
  Rpcs.deleteConversation,
  (client) => client.chat.deleteConversation
)
const StartGeneration = mutation(
  Rpcs.startGeneration,
  (client) => client.chat.startGeneration
)
const StopGeneration = mutation(
  Rpcs.stopGeneration,
  (client) => client.chat.stopGeneration
)
export const Chat = Group.make({
  ListModels,
  ListConversations,
  WatchConversation,
  CreateConversation,
  UpdateConversation,
  DeleteConversation,
  StartGeneration,
  StopGeneration,
})
