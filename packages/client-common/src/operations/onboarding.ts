import { Onboarding as Rpcs } from "@magnitudedev/sdk";
import { Group, QueryClient } from "@magnitudedev/effect-query";
import { mutation, query } from "./bind";

const GetOnboardingState = query(
  Rpcs.getOnboardingState,
  (client) => client.onboarding.getOnboardingState,
  { staleTime: Infinity, gcTime: Infinity }
);

const CompleteOnboarding = mutation(
  Rpcs.completeOnboarding,
  (client) => client.onboarding.completeOnboarding,
  { synchronize: () => QueryClient.invalidate(GetOnboardingState.match()) }
);

export const Onboarding = Group.make({
  GetOnboardingState,
  CompleteOnboarding,
});
