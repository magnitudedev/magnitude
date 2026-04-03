---
id: approve
label: Approve
description: Decision, approval, or sign-off needed from the user before proceeding.
allowedAssignees: [user]
---

<!-- @lead -->

## Inputs to present to the user
- Clear description of what needs approval and why
- Relevant context: tradeoffs, alternatives considered, recommendations
- Impact of the decision on downstream work
- Specific question or decision to be made

## Coordination loop
1. Create this task with a title that clearly describes the decision needed.
2. Assign to user and present the decision context in the assignment message.
3. Wait for user response — do not proceed on dependent work until approval is received.
4. Act on user feedback: mark complete if approved, or adjust approach based on feedback.

<!-- @criteria -->

## Completion criteria
- [ ] User has provided an explicit decision or approval.
- [ ] Lead has acted on any feedback or conditions attached to the approval.
- [ ] Dependent work is unblocked or redirected based on the decision.
