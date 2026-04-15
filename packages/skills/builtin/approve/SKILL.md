---
name: approve
description: When a decision or sign-off is needed from the user before proceeding.
---

# Approve

Decision or sign-off needed from the user before proceeding.

## Approach

Approve tasks represent decision points that only the user can resolve. Dependent work is contingent on the decision — proceeding on assumptions instead of waiting for approval risks wasted implementation if the user's answer changes direction.

This is a blocking task. Don't start dependent work until the user has responded.

## What to Present

When creating an approve task, give the user what they need to decide:

- What decision is pending and why it matters now
- The options considered, tradeoffs involved, and your recommendation if applicable
- Impact of each choice on downstream work
- A specific question to answer — not an open-ended status update

Be concrete. "Should we proceed with approach A or B?" is better than "Please review and let us know."

## Quality Bar

- User has provided an explicit decision or approval.
- Any feedback or conditions attached to the approval are acted on.
- Dependent work is unblocked or redirected based on the decision.

## Skill Evolution

Update this skill when:
- The user has preferences about how decisions should be presented.
- Certain types of decisions recur — add guidance on how to frame them.
