import { applyChatUpdate, Chat as Rpcs } from "@magnitudedev/sdk"
import { Group } from "@magnitudedev/effect-query"
import { mutation, query, streamQuery } from "./bind"

const ListChats = query(Rpcs.list, (client) => client.chat.list)
const WatchChat = streamQuery(Rpcs.watch, (client) => client.chat.watch, {
  reduce: applyChatUpdate,
  gcTime: "5 seconds",
})
const StartChat = mutation(Rpcs.start, (client) => client.chat.start)
const SendChatMessage = mutation(Rpcs.send, (client) => client.chat.send)
const RetryChatResponse = mutation(Rpcs.retry, (client) => client.chat.retry)
const StopChatResponse = mutation(Rpcs.stop, (client) => client.chat.stop)
const RenameChat = mutation(Rpcs.rename, (client) => client.chat.rename)
const RemoveChat = mutation(Rpcs.remove, (client) => client.chat.remove)

export const Chat = Group.make({
  ListChats,
  WatchChat,
  StartChat,
  SendChatMessage,
  RetryChatResponse,
  StopChatResponse,
  RenameChat,
  RemoveChat,
})
