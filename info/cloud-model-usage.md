# Cloud Model Usage

Cloud models are optional. Magnitude's open-source agent works with local
models without a cloud subscription. A Magnitude API key connects the client
to hosted models when the account has an active Pro subscription.

The provider owns subscription state, completed-cost accounting, usage windows,
and enforcement. Magnitude consumes one cloud-usage contract through the normal
SDK-to-daemon RPC path. Clients do not call the provider directly.

## Magnitude Pro

Pro costs $20 per month, with an introductory first month at $10. The discount
is configured in Autumn.

Pro includes $10 per rolling 5-hour window, $20 per rolling weekly window, and
$40 per rolling monthly window.

The provider does not reserve estimated cost. A request may finish above a
limit; following requests are denied until the violated window resets.

## Client Behavior

`/usage` and `/limits` open the same CLI view. It shows Pro subscription state,
current window usage and reset times, recent request and cost totals, model
breakdowns, and daily token activity. Without a configured Magnitude API key,
it directs the user to cloud settings instead of making the usage request.

A missing subscription is represented as a non-retryable
`SubscriptionRequired` rejection. Provider limit errors are represented as a
non-retryable `UsageLimitExceeded` rejection. Both link to the Magnitude billing
page and never suggest credit top-ups.

The provider and Magnitude must deploy together because the usage endpoint and
RPC intentionally replace the old balance contract without compatibility
aliases.
