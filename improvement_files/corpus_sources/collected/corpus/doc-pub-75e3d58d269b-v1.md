## Issues and pull requests

Mind these guidelines, the opinions of other collaborators, and guidance of the
[TSC][]. Notify other qualified parties for more input on an issue or a pull
request. See [Who to CC in the issue tracker](#who-to-cc-in-the-issue-tracker).

### Welcoming first-time contributors

Always show courtesy to individuals submitting issues and pull requests. Be
welcoming to first-time contributors, identified by the GitHub **First-time contributor** badge.

For first-time contributors, check if the commit author is the same as the pull
request author. This way, once their pull request lands, GitHub will show them
as a _Contributor_. Ask if they have configured their git
[username][git-username] and [email][git-email] to their liking.

### Closing issues and pull requests

Collaborators can close any issue or pull request that is not relevant to the
future of the Node.js project. Where this is unclear, leave the issue or pull
request open for several days to allow for discussion. Where this does not yield
evidence that the issue or pull request has relevance, close it. Remember that
issues and pull requests can always be re-opened if necessary.

### Stale issues and pull requests

The [stale workflow](../../.github/workflows/stale.yml) runs on all open issues
and pull requests. It adds the `stale` label after 90 days without activity and
closes the item after another 30 days without activity. New activity removes
the `stale` label automatically.

The `never-stale` label exempts both issues and pull requests from this
automation. The `confirmed-bug` label also exempts issues. Reserve
`never-stale` for items that need a permanent exemption. Otherwise, leave an
update when an item remains relevant or close it when it does not.

### Author ready pull requests

A pull request is _author ready_ when:

* There is a CI run in progress or completed.
* There is at least one collaborator approval.
* There are no outstanding review comments.

Please always add the `author ready` label to the pull request in that case.
Please always remove it again as soon as the conditions are not met anymore.

When approving a pull request that qualifies, add `author ready` and, if a
Jenkins CI run is required but has not started, `request-ci`. When the pull
request author is not a collaborator, it is helpful to follow the CI run through
completion and add `commit-queue` after the required CI is green.

### Handling own pull requests

When you open a pull request, [start a CI](#testing-and-ci) right away. Later,
after new code changes or rebasing, start a new CI.

As soon as the pull request is ready to land, please do so. This allows other
collaborators to focus on other pull requests. If your pull request is not ready
to land but is [author ready](#author-ready-pull-requests), add the
`author ready` label. If you wish to land the pull request yourself, use the
"assign yourself" link to self-assign it.

### Repository triage views

The repository has several pinned
[triage views](https://github.com/nodejs/node/issues/views) for managing pull
requests:

* [PR action queue](https://github.com/nodejs/node/issues/views/15196):
  Non-stale, human-authored pull requests labeled `author ready` or
  `review wanted` that are not yet in the commit queue.
* [PR attention queue](https://github.com/nodejs/node/issues/views/15058):
  Non-stale pull requests awaiting a second approval, requesting fast-track, or
  addressing flaky tests.
* [Bot PRs queue](https://github.com/nodejs/node/issues/views/15198): Open,
  non-stale Node.js GitHub Bot and Dependabot pull requests that are not yet in
  the commit queue.
* [My Active PRs](https://github.com/nodejs/node/issues/views/15142): Open pull
  requests authored by the signed-in viewer that are not yet in the commit
  queue.

Keep `author ready`, `review wanted`, `commit-queue`, and `stale` accurate so
these views remain useful.

### Managing security issues

Use the process outlined in [SECURITY.md][] to report security
issues. If a user opens a security issue in the public repository:

* Ask the user to submit a report through HackerOne as outlined in
  [SECURITY.md][].
* Move the issue to the private repository called
  [premature-disclosures](https://github.com/nodejs/premature-disclosures).
* For any related pull requests, create an associated issue in the
  `premature-disclosures` repository.  Add a copy of the patch for the
  pull request to the issue. Add screenshots of discussion from the pull request
  to the issue.
* [Open a ticket with GitHub](https://support.github.com/contact) to delete the
  pull request using Node.js (team) as the account organization.
* Open a new issue in the public repository with the title `FYI - pull request
  deleted #YYYY`. Include an explanation for the user:
  > FYI @xxxx we asked GitHub to delete your pull request while we work on
  > releases in private.
* Email `tsc@iojs.org` with links to the issues in the
  `premature-disclosures` repository.