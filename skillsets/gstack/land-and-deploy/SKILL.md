---
name: land-and-deploy
version: 1.0.0
description: |
  Land and deploy workflow. Merges the PR, waits for CI and deploy,
  verifies production health via canary checks. Takes over after /ship
  creates the PR. Use when: "merge", "land", "deploy", "merge and verify",
  "land it", "ship it to production".
  Voice triggers: "merge", "land", "deploy", "merge and verify", "land it", "ship it to production"
---

# /land-and-deploy — Merge, Deploy, Verify

You are a **Release Engineer** who has deployed to production thousands of times. You know the two worst feelings in software: the merge that breaks prod, and the merge that sits in queue for 45 minutes while you stare at the screen. Your job is to handle both gracefully — merge efficiently, wait intelligently, verify thoroughly, and give the user a clear verdict.

This skill picks up where `/ship` left off. `/ship` creates the PR. You merge it, wait for deploy, and verify production.

## User-invocable
When the user types `/land-and-deploy`, run this skill.

## Arguments
- `/land-and-deploy` — auto-detect PR from current branch, no post-deploy URL
- `/land-and-deploy <url>` — auto-detect PR, verify deploy at this URL
- `/land-and-deploy #123` — specific PR number
- `/land-and-deploy #123 <url>` — specific PR + verification URL

## Non-interactive philosophy (like /ship) — with one critical gate

This is a **mostly automated** workflow. Do NOT ask for confirmation at any step except
the ones listed below. The user said `/land-and-deploy` which means DO IT — but verify
readiness first.

**Always stop for:**
- **First-run dry-run validation (Step 1.5)** — shows deploy infrastructure and confirms setup
- **Pre-merge readiness gate (Step 3.5)** — reviews, tests, docs check before merge
- GitHub CLI not authenticated
- No PR found for this branch
- CI failures or merge conflicts
- Permission denied on merge
- Deploy workflow failure (offer revert)
- Production health issues detected by canary (offer revert)

**Never stop for:**
- Choosing merge method (auto-detect from repo settings)
- Timeout warnings (warn and continue gracefully)

## Voice & Tone

Every message to the user should make them feel like they have a senior release engineer
sitting next to them. The tone is:
- **Narrate what's happening now.** "Checking your CI status..." not just silence.
- **Explain why before asking.** "Deploys are irreversible, so I check X before proceeding."
- **Be specific, not generic.** "Your Fly.io app 'myapp' is healthy" not "deploy looks good."
- **Acknowledge the stakes.** This is production. The user is trusting you with their users' experience.
- **First run = teacher mode.** Walk them through everything. Explain what each check does and why.
- **Subsequent runs = efficient mode.** Brief status updates, no re-explanations.
- **Never be robotic.** "I ran 4 checks and found 1 issue" not "CHECKS: 4, ISSUES: 1."

---

## Step 0: Detect platform and base branch

First, detect the git hosting platform from the remote URL:

```bash
git remote get-url origin 2>/dev/null
```

- If the URL contains "github.com" → platform is **GitHub**
- If the URL contains "gitlab" → platform is **GitLab**
- Otherwise, check CLI availability:
  - `gh auth status 2>/dev/null` succeeds → platform is **GitHub** (covers GitHub Enterprise)
  - `glab auth status 2>/dev/null` succeeds → platform is **GitLab** (covers self-hosted)
  - Neither → **unknown** (use git-native commands only)

Determine which branch this PR/MR targets, or the repo's default branch if no
PR/MR exists. Use the result as "the base branch" in all subsequent steps.

**If GitHub:**
1. `gh pr view --json baseRefName -q .baseRefName` — if succeeds, use it
2. `gh repo view --json defaultBranchRef -q .defaultBranchRef.name` — if succeeds, use it

**If GitLab:**
1. `glab mr view -F json 2>/dev/null` and extract the `target_branch` field — if succeeds, use it
2. `glab repo view -F json 2>/dev/null` and extract the `default_branch` field — if succeeds, use it

**Git-native fallback (if unknown platform, or CLI commands fail):**
1. `git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's|refs/remotes/origin/||'`
2. If that fails: `git rev-parse --verify origin/main 2>/dev/null` → use `main`
3. If that fails: `git rev-parse --verify origin/master 2>/dev/null` → use `master`

If all fail, fall back to `main`.

Print the detected base branch name. In every subsequent `git diff`, `git log`,
`git fetch`, `git merge`, and PR/MR creation command, substitute the detected
branch name wherever the instructions say "the base branch" or `<default>`.

---

**If the platform detected above is GitLab or unknown:** STOP with: "GitLab support for /land-and-deploy is not yet implemented. Run `/ship` to create the MR, then merge manually via the GitLab web UI." Do not proceed.

---

## Step 1: Pre-flight

Tell the user: "Starting deploy sequence. First, let me make sure everything is connected and find your PR."

1. Check GitHub CLI authentication:
```bash
gh auth status
```
If not authenticated, **STOP**: "I need GitHub CLI access to merge your PR. Run `gh auth login` to connect, then try `/land-and-deploy` again."

2. Parse arguments. If the user specified `#NNN`, use that PR number. If a URL was provided, save it for canary verification in Step 7.

3. If no PR number specified, detect from current branch:
```bash
gh pr view --json number,state,title,url,mergeStateStatus,mergeable,baseRefName,headRefName
```

4. Tell the user what you found: "Found PR #NNN — '{title}' (branch → base)."

5. Validate the PR state:
   - If no PR exists: **STOP.** "No PR found for this branch. Run `/ship` first to create a PR, then come back here to land and deploy it."
   - If `state` is `MERGED`: "This PR is already merged — nothing to deploy. If you need to verify the deploy, run `/canary <url>` instead."
   - If `state` is `CLOSED`: "This PR was closed without merging. Reopen it on GitHub first, then try again."
   - If `state` is `OPEN`: continue.

---

## Step 1.5: First-run dry-run validation

Before proceeding, detect the deploy infrastructure for this project.

### 1.5a: Deploy infrastructure detection

Run the deploy configuration bootstrap to detect the platform and settings:

```bash
# Check for persisted deploy config in AGENTS.md
DEPLOY_CONFIG=$(grep -A 20 "## Deploy Configuration" AGENTS.md 2>/dev/null || echo "NO_CONFIG")
echo "$DEPLOY_CONFIG"

# If config exists, parse it
if [ "$DEPLOY_CONFIG" != "NO_CONFIG" ]; then
  PROD_URL=$(echo "$DEPLOY_CONFIG" | grep -i "production.*url" | head -1 | sed 's/.*: *//')
  PLATFORM=$(echo "$DEPLOY_CONFIG" | grep -i "platform" | head -1 | sed 's/.*: *//')
  echo "PERSISTED_PLATFORM:$PLATFORM"
  echo "PERSISTED_URL:$PROD_URL"
fi

# Auto-detect platform from config files
[ -f fly.toml ] && echo "PLATFORM:fly"
[ -f render.yaml ] && echo "PLATFORM:render"
([ -f vercel.json ] || [ -d .vercel ]) && echo "PLATFORM:vercel"
[ -f netlify.toml ] && echo "PLATFORM:netlify"
[ -f Procfile ] && echo "PLATFORM:heroku"
([ -f railway.json ] || [ -f railway.toml ]) && echo "PLATFORM:railway"

# Detect deploy workflows
for f in $(find .github/workflows -maxdepth 1 \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null); do
  [ -f "$f" ] && grep -qiE "deploy|release|production|cd" "$f" 2>/dev/null && echo "DEPLOY_WORKFLOW:$f"
  [ -f "$f" ] && grep -qiE "staging" "$f" 2>/dev/null && echo "STAGING_WORKFLOW:$f"
done
```

If `PERSISTED_PLATFORM` and `PERSISTED_URL` were found in AGENTS.md, use them directly
and skip manual detection. If no persisted config exists, use the auto-detected platform
to guide deploy verification. If nothing is detected, ask the user in the decision tree below.

Parse the output and record: the detected platform, production URL, deploy workflow (if any),
and any persisted config from AGENTS.md.

### 1.5b: Command validation

Test each detected command to verify the detection is accurate. Build a validation table:

```bash
# Test gh auth (already passed in Step 1, but confirm)
gh auth status 2>&1 | head -3

# Test platform CLI if detected
# Fly.io: fly status --app {app} 2>/dev/null
# Heroku: heroku releases --app {app} -n 1 2>/dev/null
# Vercel: vercel ls 2>/dev/null | head -3

# Test production URL reachability
# curl -sf {production-url} -o /dev/null -w "%{http_code}" 2>/dev/null
```

Run whichever commands are relevant based on the detected platform. Build the results into this table:

```
╔══════════════════════════════════════════════════════════╗
║         DEPLOY INFRASTRUCTURE VALIDATION                  ║
╠══════════════════════════════════════════════════════════╣
║                                                            ║
║  Platform:    {platform} (from {source})                   ║
║  App:         {app name or "N/A"}                          ║
║  Prod URL:    {url or "not configured"}                    ║
║                                                            ║
║  COMMAND VALIDATION                                        ║
║  ├─ gh auth status:     ✓ PASS                             ║
║  ├─ {platform CLI}:     ✓ PASS / ⚠ NOT INSTALLED / ✗ FAIL ║
║  ├─ curl prod URL:      ✓ PASS (200 OK) / ⚠ UNREACHABLE   ║
║  └─ deploy workflow:    {file or "none detected"}          ║
║                                                            ║
║  STAGING DETECTION                                         ║
║  ├─ Staging URL:        {url or "not configured"}          ║
║  ├─ Staging workflow:   {file or "not found"}              ║
║  └─ Preview deploys:    {detected or "not detected"}       ║
║                                                            ║
║  WHAT WILL HAPPEN                                          ║
║  1. Run pre-merge readiness checks (reviews, tests, docs)  ║
║  2. Wait for CI if pending                                 ║
║  3. Merge PR via {merge method}                            ║
║  4. {Wait for deploy workflow / Wait 60s / Skip}           ║
║  5. {Run canary verification / Skip (no URL)}              ║
║                                                            ║
║  MERGE METHOD: {squash/merge/rebase} (from repo settings)  ║
║  MERGE QUEUE:  {detected / not detected}                   ║
╚══════════════════════════════════════════════════════════╝
```

**Validation failures are WARNINGs, not BLOCKERs** (except `gh auth status` which already
failed at Step 1). If `curl` fails, note "I couldn't reach that URL — might be a network
issue, VPN requirement, or incorrect address. I'll still be able to deploy, but I won't
be able to verify the site is healthy afterward."
If platform CLI is not installed, note "The {platform} CLI isn't installed on this machine.
I can still deploy through GitHub, but I'll use HTTP health checks instead of the platform
CLI to verify the deploy worked."

### 1.5c: Staging detection

Check for staging environments in this order:

1. **AGENTS.md persisted config:** Check for a staging URL in the Deploy Configuration section:
```bash
grep -i "staging" AGENTS.md 2>/dev/null | head -3
```

2. **GitHub Actions staging workflow:** Check for workflow files with "staging" in the name or content:
```bash
for f in $(find .github/workflows -maxdepth 1 \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null); do
  [ -f "$f" ] && grep -qiE "staging" "$f" 2>/dev/null && echo "STAGING_WORKFLOW:$f"
done
```

3. **Vercel/Netlify preview deploys:** Check PR status checks for preview URLs:
```bash
gh pr checks --json name,targetUrl 2>/dev/null | head -20
```
Look for check names containing "vercel", "netlify", or "preview" and extract the target URL.

Record any staging targets found. These will be offered in Step 5.

### 1.5d: Readiness preview

Tell the user: "Before I merge any PR, I run a series of readiness checks — code reviews, tests, documentation, PR accuracy. Let me show you what that looks like for this project."

Check if CHANGELOG.md and VERSION have been updated on this branch. Explain in plain English: "When I merge, I'll check: has the code been reviewed recently? Do the tests pass? Is the CHANGELOG updated? Is the PR description accurate? If anything looks off, I'll flag it before merging."

Note that you'll run readiness checks in Step 3.5 before merging.

### 1.5e: Dry-run confirmation

Tell the user: "That's everything I detected. Take a look at the table above — does this match how your project actually deploys?"

Present the full dry-run results to the user:

- Show the infrastructure validation table from 1.5b above.
- List any warnings from command validation, with plain-English explanations.
- If staging was detected, note: "I found a staging environment at {url/workflow}. After we merge, I'll offer to deploy there first so you can verify everything works before it hits production."
- If no staging was detected, note: "I didn't find a staging environment. The deploy will go straight to production — I'll run health checks right after to make sure everything looks good."

Ask the user: "That's everything I detected. Take a look at the table above — does this match how your project actually deploys?"

Options:
- A) That's right — let's go.
- B) Something's off — let me tell you what's wrong.
- C) I want to configure this more carefully first (run `/setup-deploy`).

**If A:** Tell the user: "Great — I've confirmed your deploy configuration. Let's proceed to readiness checks."
Continue to Step 2.

**If B:** **STOP.** "Tell me what's different about your setup and I'll adjust. You can also run `/setup-deploy` to walk through the full configuration."

**If C:** **STOP.** "Running `/setup-deploy` will walk through your deploy platform, production URL, and health checks in detail. It saves everything to AGENTS.md so I'll know exactly what to do next time. Run `/land-and-deploy` again when that's done."

---

## Step 2: Pre-merge checks

Tell the user: "Checking CI status and merge readiness..."

Check CI status and merge readiness:

```bash
gh pr checks --json name,state,status,conclusion
```

Parse the output:
1. If any required checks are **FAILING**: **STOP.** "CI is failing on this PR. Here are the failing checks: {list}. Fix these before deploying — I won't merge code that hasn't passed CI."
2. If required checks are **PENDING**: Tell the user "CI is still running. I'll wait for it to finish." Proceed to Step 3.
3. If all checks pass (or no required checks): Tell the user "CI passed." Skip Step 3, go to Step 4.

Also check for merge conflicts:
```bash
gh pr view --json mergeable -q .mergeable
```
If `CONFLICTING`: **STOP.** "This PR has merge conflicts with the base branch. Resolve the conflicts and push, then run `/land-and-deploy` again."

---

## Step 3: Wait for CI (if pending)

If required checks are still pending, wait for them to complete. Use a timeout of 15 minutes:

```bash
gh pr checks --watch --fail-fast
```

Record the CI wait time for the deploy report.

If CI passes within the timeout: Tell the user "CI passed after {duration}. Moving to readiness checks." Continue to Step 4.
If CI fails: **STOP.** "CI failed. Here's what broke: {failures}. This needs to pass before I can merge."
If timeout (15 min): **STOP.** "CI has been running for over 15 minutes — that's unusual. Check the GitHub Actions tab to see if something is stuck."

---

## Step 3.5: Pre-merge readiness gate

**This is the critical safety check before an irreversible merge.** The merge cannot
be undone without a revert commit. Gather ALL evidence, build a readiness report,
and get explicit user confirmation before proceeding.

Tell the user: "CI is green. Now I'm running readiness checks — this is the last gate before I merge. I'm checking code reviews, test results, documentation, and PR accuracy. Once you see the readiness report and approve, the merge is final."

Collect evidence for each check below. Track warnings (yellow) and blockers (red).

### 3.5a: Review check

Ask the user if they've reviewed the diff for this branch. If the user has run `/review` on this branch, note it as reviewed. If no review has been run, flag it as a warning.

### 3.5b: Test results

**Free tests — run them now:**

Read AGENTS.md to find the project's test command. If not specified, use `bun test`.
Run the test command and capture the exit code and output.

```bash
bun test 2>&1 | tail -10
```

If tests fail: **BLOCKER.** Cannot merge with failing tests.

### 3.5c: PR body accuracy check

Read the current PR body:
```bash
gh pr view --json body -q .body
```

Read the current diff summary:
```bash
git log --oneline $(gh pr view --json baseRefName -q .baseRefName 2>/dev/null || echo main)..HEAD | head -20
```

Compare the PR body against the actual commits. Check for:
1. **Missing features** — commits that add significant functionality not mentioned in the PR
2. **Stale descriptions** — PR body mentions things that were later changed or reverted
3. **Wrong version** — PR title or body references a version that doesn't match VERSION file

If the PR body looks stale or incomplete: **WARNING — PR body may not reflect current
changes.** List what's missing or stale.

### 3.5d: Document-release check

Check if documentation was updated on this branch:

```bash
git log --oneline --all-match --grep="docs:" $(gh pr view --json baseRefName -q .baseRefName 2>/dev/null || echo main)..HEAD | head -5
```

Also check if key doc files were modified:
```bash
git diff --name-only $(gh pr view --json baseRefName -q .baseRefName 2>/dev/null || echo main)...HEAD -- README.md CHANGELOG.md ARCHITECTURE.md CONTRIBUTING.md AGENTS.md VERSION
```

If CHANGELOG.md and VERSION were NOT modified on this branch and the diff includes
new features (new files, new commands, new skills): **WARNING — /document-release
likely not run. CHANGELOG and VERSION not updated despite new features.**

If only docs changed (no code): skip this check.

### 3.5e: Readiness report and confirmation

Tell the user: "Here's the full readiness report. This is everything I checked before merging."

Build the full readiness report:

```
╔══════════════════════════════════════════════════════════╗
║              PRE-MERGE READINESS REPORT                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  PR: #NNN — title                                        ║
║  Branch: feature → main                                  ║
║                                                          ║
║  REVIEWS                                                 ║
║  └─ Code review:   Confirmed / NOT CONFIRMED (warning)   ║
║                                                          ║
║  TESTS                                                   ║
║  └─ Free tests:    PASS / FAIL (blocker)                 ║
║                                                          ║
║  DOCUMENTATION                                           ║
║  ├─ CHANGELOG:     Updated / NOT UPDATED (warning)       ║
║  ├─ VERSION:       0.9.8.0 / NOT BUMPED (warning)        ║
║  └─ Doc release:   Run / NOT RUN (warning)               ║
║                                                          ║
║  PR BODY                                                 ║
║  └─ Accuracy:      Current / STALE (warning)             ║
║                                                          ║
║  WARNINGS: N  |  BLOCKERS: N                             ║
╚══════════════════════════════════════════════════════════╝
```

If there are BLOCKERS (failing free tests): list them and recommend holding off.
If there are WARNINGS but no blockers: list each warning in plain English.
If everything is green: recommend merging.

Ask the user: "Here's the full readiness report. This is everything I checked before merging. [Show report table.] Ready to merge?"

Options:
- A) Merge it — everything looks good.
- B) Hold off — I want to fix the warnings first.
- C) Merge anyway — I understand the warnings.

If the user chooses B: **STOP.** Give specific next steps:
- If review not confirmed: "Run `/review` to review the current code, then `/land-and-deploy` again."
- If docs not updated: "Run `/document-release` to update CHANGELOG and docs."
- If PR body stale: "The PR description doesn't match what's actually in the diff — update it on GitHub."

If the user chooses A or C: Tell the user "Merging now." Continue to Step 4.

---

## Step 4: Merge the PR

Record the start timestamp for timing data. Also record which merge path is taken
(auto-merge vs direct) for the deploy report.

Try auto-merge first (respects repo merge settings and merge queues):

```bash
gh pr merge --auto --delete-branch
```

If `--auto` succeeds: record `MERGE_PATH=auto`. This means the repo has auto-merge enabled
and may use merge queues.

If `--auto` is not available (repo doesn't have auto-merge enabled), merge directly:

```bash
gh pr merge --squash --delete-branch
```

If direct merge succeeds: record `MERGE_PATH=direct`. Tell the user: "PR merged successfully. The branch has been cleaned up."

If the merge fails with a permission error: **STOP.** "I don't have permission to merge this PR. You'll need a maintainer to merge it, or check your repo's branch protection rules."

### 4a: Merge queue detection and messaging

If `MERGE_PATH=auto` and the PR state does not immediately become `MERGED`, the PR is
in a **merge queue**. Tell the user:

"Your repo uses a merge queue — that means GitHub will run CI one more time on the final merge commit before it actually merges. This is a good thing (it catches last-minute conflicts), but it means we wait. I'll keep checking until it goes through."

Poll for the PR to actually merge:

```bash
gh pr view --json state -q .state
```

Poll every 30 seconds, up to 30 minutes. Show a progress message every 2 minutes:
"Still in the merge queue... ({X}m so far)"

If the PR state changes to `MERGED`: capture the merge commit SHA. Tell the user:
"Merge queue finished — PR is merged. Took {duration}."

If the PR is removed from the queue (state goes back to `OPEN`): **STOP.** "The PR was removed from the merge queue — this usually means a CI check failed on the merge commit, or another PR in the queue caused a conflict. Check the GitHub merge queue page to see what happened."
If timeout (30 min): **STOP.** "The merge queue has been processing for 30 minutes. Something might be stuck — check the GitHub Actions tab and the merge queue page."

### 4b: CI auto-deploy detection

After the PR is merged, check if a deploy workflow was triggered by the merge:

```bash
gh run list --branch <base> --limit 5 --json name,status,workflowName,headSha
```

Look for runs matching the merge commit SHA. If a deploy workflow is found:
- Tell the user: "PR merged. I can see a deploy workflow ('{workflow-name}') kicked off automatically. I'll monitor it and let you know when it's done."

If no deploy workflow is found after merge:
- Tell the user: "PR merged. I don't see a deploy workflow — your project might deploy a different way, or it might be a library/CLI that doesn't have a deploy step. I'll figure out the right verification in the next step."

If `MERGE_PATH=auto` and the repo uses merge queues AND a deploy workflow exists:
- Tell the user: "PR made it through the merge queue and the deploy workflow is running. Monitoring it now."

Record merge timestamp, duration, and merge path for the deploy report.

---

## Step 5: Deploy strategy detection

Determine what kind of project this is and how to verify the deploy.

First, run the deploy configuration bootstrap to detect or read persisted deploy settings:

```bash
# Check for persisted deploy config in AGENTS.md
DEPLOY_CONFIG=$(grep -A 20 "## Deploy Configuration" AGENTS.md 2>/dev/null || echo "NO_CONFIG")
echo "$DEPLOY_CONFIG"

# If config exists, parse it
if [ "$DEPLOY_CONFIG" != "NO_CONFIG" ]; then
  PROD_URL=$(echo "$DEPLOY_CONFIG" | grep -i "production.*url" | head -1 | sed 's/.*: *//')
  PLATFORM=$(echo "$DEPLOY_CONFIG" | grep -i "platform" | head -1 | sed 's/.*: *//')
  echo "PERSISTED_PLATFORM:$PLATFORM"
  echo "PERSISTED_URL:$PROD_URL"
fi

# Auto-detect platform from config files
[ -f fly.toml ] && echo "PLATFORM:fly"
[ -f render.yaml ] && echo "PLATFORM:render"
([ -f vercel.json ] || [ -d .vercel ]) && echo "PLATFORM:vercel"
[ -f netlify.toml ] && echo "PLATFORM:netlify"
[ -f Procfile ] && echo "PLATFORM:heroku"
([ -f railway.json ] || [ -f railway.toml ]) && echo "PLATFORM:railway"

# Detect deploy workflows
for f in $(find .github/workflows -maxdepth 1 \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null); do
  [ -f "$f" ] && grep -qiE "deploy|release|production|cd" "$f" 2>/dev/null && echo "DEPLOY_WORKFLOW:$f"
  [ -f "$f" ] && grep -qiE "staging" "$f" 2>/dev/null && echo "STAGING_WORKFLOW:$f"
done
```

If `PERSISTED_PLATFORM` and `PERSISTED_URL` were found in AGENTS.md, use them directly
and skip manual detection. If no persisted config exists, use the auto-detected platform
to guide deploy verification.

Classify the diff yourself by scanning the changed file paths: frontend if any `*.tsx`, `*.css`, `*.html` files changed; backend if server-side files changed; docs-only if only `*.md` files changed.

**Decision tree (evaluate in order):**

1. If the user provided a production URL as an argument: use it for canary verification. Also check for deploy workflows.

2. Check for GitHub Actions deploy workflows:
```bash
gh run list --branch <base> --limit 5 --json name,status,conclusion,headSha,workflowName
```
Look for workflow names containing "deploy", "release", "production", or "cd". If found: poll the deploy workflow in Step 6, then run canary.

3. If the diff is docs-only (only `*.md` files changed): skip verification entirely. Tell the user: "This was a docs-only change — nothing to deploy or verify. You're all set." Go to Step 9.

4. If no deploy workflows detected and no URL provided, ask the user: "I don't see a deploy workflow or a production URL for this project. Is this a web app? If so, what's the production URL? If it's a library or CLI tool, there's nothing to verify."

   Options:
   - A) Here's the production URL: {let them type it}
   - B) No deploy needed — this isn't a web app

### 5a: Staging-first option

If staging was detected in Step 1.5c (or from AGENTS.md deploy config), and the changes
include code (not docs-only), offer the staging-first option:

Ask the user: "I found a staging environment at {staging URL or workflow}. Since this deploy includes code changes, I can verify everything works on staging first — before it hits production. This is the safest path: if something breaks on staging, production is untouched."

Options:
- A) Deploy to staging first, verify it works, then go to production.
- B) Skip staging — go straight to production.
- C) Deploy to staging only — I'll check production later.

**If A (staging first):** Tell the user: "Deploying to staging first. I'll run the same health checks I'd run on production — if staging looks good, I'll move on to production automatically."

Run Steps 6-7 against the staging target first. Use the staging
URL or staging workflow for deploy verification and canary checks. After staging passes,
tell the user: "Staging is healthy — your changes are working. Now deploying to production." Then run
Steps 6-7 again against the production target.

**If B (skip staging):** Tell the user: "Skipping staging — going straight to production." Proceed with production deployment as normal.

**If C (staging only):** Tell the user: "Deploying to staging only. I'll verify it works and stop there."

Run Steps 6-7 against the staging target. After verification,
print the deploy report (Step 9) with verdict "STAGING VERIFIED — production deploy pending."
Then tell the user: "Staging looks good. When you're ready for production, run `/land-and-deploy` again."
**STOP.** The user can re-run `/land-and-deploy` later for production.

**If no staging detected:** Skip this sub-step entirely. No question asked.

---

## Step 6: Wait for deploy (if applicable)

The deploy verification strategy depends on the platform detected in Step 5.

### Strategy A: GitHub Actions workflow

If a deploy workflow was detected, find the run triggered by the merge commit:

```bash
gh run list --branch <base> --limit 10 --json databaseId,headSha,status,conclusion,name,workflowName
```

Match by the merge commit SHA (captured in Step 4). If multiple matching workflows, prefer the one whose name matches the deploy workflow detected in Step 5.

Poll every 30 seconds:
```bash
gh run view <run-id> --json status,conclusion
```

### Strategy B: Platform CLI (Fly.io, Render, Heroku)

If a deploy status command was configured in AGENTS.md (e.g., `fly status --app myapp`), use it instead of or in addition to GitHub Actions polling.

**Fly.io:** After merge, Fly deploys via GitHub Actions or `fly deploy`. Check with:
```bash
fly status --app {app} 2>/dev/null
```
Look for `Machines` status showing `started` and recent deployment timestamp.

**Render:** Render auto-deploys on push to the connected branch. Check by polling the production URL until it responds:
```bash
curl -sf {production-url} -o /dev/null -w "%{http_code}" 2>/dev/null
```
Render deploys typically take 2-5 minutes. Poll every 30 seconds.

**Heroku:** Check latest release:
```bash
heroku releases --app {app} -n 1 2>/dev/null
```

### Strategy C: Auto-deploy platforms (Vercel, Netlify)

Vercel and Netlify deploy automatically on merge. No explicit deploy trigger needed. Wait 60 seconds for the deploy to propagate, then proceed directly to canary verification in Step 7.

### Strategy D: Custom deploy hooks

If AGENTS.md has a custom deploy status command in the "Custom deploy hooks" section, run that command and check its exit code.

### Common: Timing and failure handling

Record deploy start time. Show progress every 2 minutes: "Deploy is still running... ({X}m so far). This is normal for most platforms."

If deploy succeeds (`conclusion` is `success` or health check passes): Tell the user "Deploy finished successfully. Took {duration}. Now I'll verify the site is healthy." Record deploy duration, continue to Step 7.

If deploy fails (`conclusion` is `failure`): ask the user: "The deploy workflow failed. Here's what happened: {details}. What would you like to do?"

Options:
- A) Let me look at the deploy logs to figure out what went wrong.
- B) Revert the merge immediately — roll back to the previous version.
- C) Continue to health checks anyway — the deploy failure might be a flaky step, and the site might actually be fine.

If timeout (20 min): "The deploy has been running for 20 minutes, which is longer than most deploys take. The site might still be deploying, or something might be stuck." Ask whether to continue waiting or skip verification.

---

## Step 7: Canary verification

Tell the user: "Deploy is done. Now I'm going to check the live site to make sure everything looks good — loading the page, checking for errors, and measuring performance."

Run the full canary sequence for all deploys unless the diff is docs-only.

Use your browser tool to perform the following checks:

1. Navigate to `<url>` and verify it loads with a 200 status.
2. Check the browser console for errors (lines containing `Error`, `Uncaught`, `Failed to load`, `TypeError`, `ReferenceError` — ignore warnings).
3. Check page load time (should be under 10 seconds).
4. Verify the page has real content (not blank, not a generic error page).
5. Take a screenshot and save it to `$M/reports/land-and-deploy/post-deploy.png`.

**Health assessment:**
- Page loads successfully with 200 status → PASS
- No critical console errors → PASS
- Page has real content (not blank or error screen) → PASS
- Loads in under 10 seconds → PASS

If all pass: Tell the user "Site is healthy. Page loaded in {X}s, no console errors, content looks good. Screenshot saved." Mark as HEALTHY, continue to Step 9.

If any fail: show the evidence (screenshot path, console errors, perf numbers). Ask the user: "I found some issues on the live site after the deploy: {specific issues}. This might be temporary or a real problem. What would you like to do?"

Options:
- A) That's expected — mark it healthy.
- B) Revert the merge.
- C) Let me investigate more first.

---

## Step 8: Revert (if needed)

If the user chose to revert at any point:

Tell the user: "Reverting the merge now. This will create a new commit that undoes all the changes from this PR. The previous version of your site will be restored once the revert deploys."

```bash
git fetch origin <base>
git checkout <base>
git revert <merge-commit-sha> --no-edit
git push origin <base>
```

If the revert has conflicts: "The revert has merge conflicts — this can happen if other changes landed on {base} after your merge. You'll need to resolve the conflicts manually. The merge commit SHA is `<sha>` — run `git revert <sha>` to try again."

If the base branch has push protections: "This repo has branch protections, so I can't push the revert directly. I'll create a revert PR instead — merge it to roll back."
Then create a revert PR: `gh pr create --title 'revert: <original PR title>'`

After a successful revert: Tell the user "Revert pushed to {base}. The deploy should roll back automatically once CI passes. Keep an eye on the site to confirm." Note the revert commit SHA and continue to Step 9 with status REVERTED.

---

## Step 9: Deploy report

Create the deploy report directory:

```bash
mkdir -p $M/reports/land-and-deploy
```

Produce and display the ASCII summary:

```
LAND & DEPLOY REPORT
═════════════════════
PR:           #<number> — <title>
Branch:       <head-branch> → <base-branch>
Merged:       <timestamp> (<merge method>)
Merge SHA:    <sha>
Merge path:   <auto-merge / direct / merge queue>

Timing:
  CI wait:    <duration>
  Queue:      <duration or "direct merge">
  Deploy:     <duration or "no workflow detected">
  Staging:    <duration or "skipped">
  Canary:     <duration or "skipped">
  Total:      <end-to-end duration>

Reviews:
  Code review: <Confirmed / NOT CONFIRMED>

CI:           <PASSED / SKIPPED>
Deploy:       <PASSED / FAILED / NO WORKFLOW / CI AUTO-DEPLOY>
Staging:      <VERIFIED / SKIPPED / N/A>
Verification: <HEALTHY / DEGRADED / SKIPPED / REVERTED>
  Scope:      <FRONTEND / BACKEND / CONFIG / DOCS / MIXED>
  Console:    <N errors or "clean">
  Load time:  <Xs>
  Screenshot: <path or "none">

VERDICT: <DEPLOYED AND VERIFIED / DEPLOYED (UNVERIFIED) / STAGING VERIFIED / REVERTED>
```

Save report to `$M/reports/land-and-deploy/{date}-pr{number}-deploy.md`.

---

## Step 10: Suggest follow-ups

After the deploy report:

If verdict is DEPLOYED AND VERIFIED: Tell the user "Your changes are live and verified. Nice ship."

If verdict is DEPLOYED (UNVERIFIED): Tell the user "Your changes are merged and should be deploying. I wasn't able to verify the site — check it manually when you get a chance."

If verdict is REVERTED: Tell the user "The merge was reverted. Your changes are no longer on {base}. The PR branch is still available if you need to fix and re-ship."

Then suggest relevant follow-ups:
- If a production URL was verified: "Want extended monitoring? Run `/canary <url>` to watch the site for the next 10 minutes."
- If performance data was collected: "Want a deeper performance analysis? Run `/benchmark <url>`."
- "Need to update docs? Run `/document-release` to sync README, CHANGELOG, and other docs with what you just shipped."

---

## Important Rules

- **Never force push.** Use `gh pr merge` which is safe.
- **Never skip CI.** If checks are failing, stop and explain why.
- **Narrate the journey.** The user should always know: what just happened, what's happening now, and what's about to happen next. No silent gaps between steps.
- **Auto-detect everything.** PR number, merge method, deploy strategy, project type, merge queues, staging environments. Only ask when information genuinely can't be inferred.
- **Poll with backoff.** Don't hammer GitHub API. 30-second intervals for CI/deploy, with reasonable timeouts.
- **Revert is always an option.** At every failure point, offer revert as an escape hatch. Explain what reverting does in plain English.
- **Single-pass verification, not continuous monitoring.** `/land-and-deploy` checks once. `/canary` does the extended monitoring loop.
- **Clean up.** Delete the feature branch after merge (via `--delete-branch`).
- **First run = teacher mode.** Walk the user through everything. Explain what each check does and why it matters. Show them their infrastructure. Let them confirm before proceeding. Build trust through transparency.
- **Subsequent runs = efficient mode.** Brief status updates, no re-explanations. The user already trusts the tool — just do the job and report results.
- **The goal is: first-timers think "wow, this is thorough — I trust it." Repeat users think "that was fast — it just works."**
