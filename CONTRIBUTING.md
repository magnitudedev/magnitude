# Contributing to Magnitude

Contributions should be conducted through GitHub Issues and PRs.

It is expected that contributions involve AI-generated content. However, to facilate useful discussion, issues and PRs should include human-written descriptions. Ideally, designate human and AI written sections clearly.

Issues and PRs should be created to address a real, user-facing suggestion or concern - not arbitrary code quality or internal tooling problems, unless those actually affect you as a contributor and are pertinent to a contribution you are making.

If a PR involves an implementation that is one-shot by an agent with no meaningful human steering or validation, it is preferrable to submit an Issue instead.

Any PR should involve:
- Clear justification for the change and the implemented approach
- A corresponding patch changeset with a brief single-line descriptor
- A human written description somewhere, AI written portion optional
- Implementation that is aligned with codebase patterns and high quality
- Explanation of testing conducted - preferably "AI-manual" or even better "Human-manual" rather than strawman unit tests
- Do not include unit tests that do not demonstrate a meaningful property of the system

No particular domains of the codebase are necessarily on or off-limits for contributions, but targetted, clearly scoped changes that solve a specific bug are most likely to be accepted.
