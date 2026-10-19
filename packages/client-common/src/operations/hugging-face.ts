import { HuggingFace as Rpcs } from "@magnitudedev/sdk";
import { Group, QueryClient } from "@magnitudedev/effect-query";
import { mutation, query } from "./bind";

/** The Hugging Face account ACN signs in to; ACN pokes it on every change, including a sign-in completing elsewhere. */
const GetHuggingFaceAccount = query(Rpcs.getHuggingFaceAccount, (client) => client.huggingFace.getHuggingFaceAccount);
const synchronize = () => QueryClient.invalidate(GetHuggingFaceAccount.match({}));

const StartHuggingFaceSignIn = mutation(Rpcs.startHuggingFaceSignIn, (client) => client.huggingFace.startHuggingFaceSignIn, { synchronize });
const StartHuggingFaceDeviceSignIn = mutation(Rpcs.startHuggingFaceDeviceSignIn, (client) => client.huggingFace.startHuggingFaceDeviceSignIn, { synchronize });
const CancelHuggingFaceSignIn = mutation(Rpcs.cancelHuggingFaceSignIn, (client) => client.huggingFace.cancelHuggingFaceSignIn, { synchronize });
const SignOutOfHuggingFace = mutation(Rpcs.signOutOfHuggingFace, (client) => client.huggingFace.signOutOfHuggingFace, { synchronize });

export const HuggingFace = Group.make({
  GetHuggingFaceAccount,
  StartHuggingFaceSignIn,
  StartHuggingFaceDeviceSignIn,
  CancelHuggingFaceSignIn,
  SignOutOfHuggingFace,
});
