import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { replaySafe } from "../transport/recovery"
import {
  ChatCommandId,
  ChatConversation,
  ChatConversationId,
  ChatFailed,
  ChatGenerationId,
  ChatHistory,
  ChatMessageId,
  ChatModels,
  ChatSettings,
  ChatUpdate,
} from "../schemas/chat"
const conversation = Schema.Struct({ conversationId: ChatConversationId })
export const Chat = {
  listModels: Rpc.make("ListChatModels", {
    payload: Schema.Struct({}),
    success: ChatModels,
    error: ChatFailed,
  }).pipe(replaySafe),
  listConversations: Rpc.make("ListChatConversations", {
    payload: Schema.Struct({
      offset: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
      limit: Schema.Number.pipe(Schema.int(), Schema.between(1, 100)),
    }),
    success: ChatHistory,
    error: ChatFailed,
  }).pipe(replaySafe),
  getConversation: Rpc.make("GetChatConversation", {
    payload: conversation,
    success: ChatConversation,
    error: ChatFailed,
  }).pipe(replaySafe),
  createConversation: Rpc.make("CreateChatConversation", {
    payload: Schema.Struct({
      conversationId: ChatConversationId,
      settings: ChatSettings,
    }),
    success: ChatConversation,
    error: ChatFailed,
  }).pipe(replaySafe),
  updateConversation: Rpc.make("UpdateChatConversation", {
    payload: Schema.Struct({
      conversationId: ChatConversationId,
      title: Schema.optionalWith(
        Schema.String.pipe(Schema.minLength(1), Schema.maxLength(200)),
        { as: "Option", exact: true }
      ),
      settings: Schema.optionalWith(ChatSettings, {
        as: "Option",
        exact: true,
      }),
    }),
    success: ChatConversation,
    error: ChatFailed,
  }).pipe(replaySafe),
  deleteConversation: Rpc.make("DeleteChatConversation", {
    payload: conversation,
    success: Schema.Void,
    error: ChatFailed,
  }).pipe(replaySafe),
  startGeneration: Rpc.make("StartChatGeneration", {
    payload: Schema.Struct({
      conversationId: ChatConversationId,
      commandId: ChatCommandId,
      content: Schema.optionalWith(
        Schema.String.pipe(Schema.minLength(1), Schema.maxLength(1_048_576)),
        { as: "Option", exact: true }
      ),
      retryMessageId: Schema.optionalWith(ChatMessageId, {
        as: "Option",
        exact: true,
      }),
      settings: ChatSettings,
    }),
    success: ChatGenerationId,
    error: ChatFailed,
  }).pipe(replaySafe),
  stopGeneration: Rpc.make("StopChatGeneration", {
    payload: Schema.Struct({
      conversationId: ChatConversationId,
      generationId: ChatGenerationId,
    }),
    success: Schema.Void,
    error: ChatFailed,
  }).pipe(replaySafe),
  watchConversation: Rpc.make("WatchChatConversation", {
    payload: conversation,
    success: ChatUpdate,
    error: ChatFailed,
    stream: true,
  }),
}
