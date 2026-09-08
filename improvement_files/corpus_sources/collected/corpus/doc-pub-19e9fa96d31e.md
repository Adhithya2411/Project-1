## Labels

### General labels

* `confirmed-bug`: Bugs you have verified
* `commit-queue`: Pull requests queued for automated landing. See the
  [commit queue guide][commit-queue.md]
* `discuss`: Things that need larger discussion
* `fast-track`: PRs that need to land faster - see
  [Waiting for approvals](#waiting-for-approvals)
* `feature request`: Any issue that requests a new feature
* `good first issue`: Issues suitable for newcomers to fix
* `lacks-second-approval`: An automatically managed label for queued pull
  requests awaiting another approval or completion of the required wait
* `meta`: Governance, policies, procedures, etc.
* `needs-ci`: Pull requests that require a full Jenkins CI run. See
  [Testing and CI](#testing-and-ci)
* `never-stale`: Issues and pull requests exempt from automatic stale handling
* `request-ci`: When this label is added to a PR, CI will be started
  automatically. See [Starting a Jenkins CI job](#starting-a-jenkins-ci-job)
* `stale`: Issues and pull requests with no activity for 90 days. See
  [Stale issues and pull requests](#stale-issues-and-pull-requests)
* `tsc-agenda`: Open issues and pull requests with this label will be added to
  the Technical Steering Committee meeting agenda

***

* `author ready` - A pull request is _author ready_ when:
  * There is a CI run in progress or completed.
  * There is at least one collaborator approval (or two TSC approvals for
    semver-major pull requests).
  * There are no outstanding review comments.

Please always add the `author ready` label to pull requests that qualify.
Please always remove it again as soon as the conditions are not met anymore,
such as if the CI run fails or a new outstanding review comment is posted.

***

* `semver-{minor,major}`
  * be conservative – that is, if a change has the remote _chance_ of breaking
    something, go for semver-major
  * when adding a semver label, add a comment explaining why you're adding it
  * minor vs. patch: roughly: "does it add a new method / does it add a new
    section to the docs"
  * major vs. everything else: run last versions tests against this version, if
    they pass, **probably** minor or patch

### LTS/version labels

We use labels to keep track of which branches a commit should land on:

* `dont-land-on-v?.x`
  * For changes that do not apply to a certain release line
  * Also used when the work of backporting a change outweighs the benefits
* `land-on-v?.x`
  * Used by releasers to mark a pull request as scheduled for inclusion in an
    LTS release
  * Applied to the original pull request for clean cherry-picks, to the backport
    pull request otherwise
* `backport-requested-v?.x`
  * Used to indicate that a pull request needs a manual backport to a branch in
    order to land the changes on that branch
  * Typically applied by a releaser when the pull request does not apply cleanly
    or it breaks the tests after applying
  * Will be replaced by either `dont-land-on-v?.x` or `backported-to-v?.x`
* `backported-to-v?.x`
  * Applied to pull requests for which a backport pull request has been merged
* `lts-watch-v?.x`
  * Applied to pull requests which the Release working group should consider
    including in an LTS release
  * Does not indicate that any specific action will be taken, but can be
    effective as messaging to non-collaborators
* `release-agenda`
  * For things that need discussion by the Release working group
  * (for example semver-minor changes that need or should go into an LTS
    release)
* `v?.x`
  * Automatically applied to changes that do not target `main` but rather the
    `v?.x-staging` branch

Once a release line enters maintenance mode, the corresponding labels do not
need to be attached anymore, as only important bugfixes will be included.

### Other labels

* Operating system labels
  * `macos`, `windows`, `smartos`, `aix`, `linux`, etc.
* Architecture labels
  * `arm`, `mips`, `s390`, `ppc`, etc.
  * No `x86{_64}` label because it is the implied default

["Merge pull request"]: https://help.github.com/articles/merging-a-pull-request/#merging-a-pull-request-on-github
[@nodejs/V8]: https://github.com/orgs/nodejs/teams/V8
[@nodejs/assert]: https://github.com/orgs/nodejs/teams/assert
[@nodejs/async_hooks]: https://github.com/orgs/nodejs/teams/async_hooks
[@nodejs/benchmarking]: https://github.com/orgs/nodejs/teams/benchmarking
[@nodejs/buffer]: https://github.com/orgs/nodejs/teams/buffer
[@nodejs/build]: https://github.com/orgs/nodejs/teams/build
[@nodejs/child_process]: https://github.com/orgs/nodejs/teams/child_process
[@nodejs/cluster]: https://github.com/orgs/nodejs/teams/cluster
[@nodejs/crypto]: https://github.com/orgs/nodejs/teams/crypto
[@nodejs/delivery-channels]: https://github.com/orgs/nodejs/teams/delivery-channels
[@nodejs/dgram]: https://github.com/orgs/nodejs/teams/dgram
[@nodejs/diagnostics]: https://github.com/orgs/nodejs/teams/diagnostics
[@nodejs/documentation]: https://github.com/orgs/nodejs/teams/documentation
[@nodejs/domains]: https://github.com/orgs/nodejs/teams/domains
[@nodejs/fs]: https://github.com/orgs/nodejs/teams/fs
[@nodejs/gyp]: https://github.com/orgs/nodejs/teams/gyp
[@nodejs/http]: https://github.com/orgs/nodejs/teams/http
[@nodejs/http2]: https://github.com/orgs/nodejs/teams/http2
[@nodejs/libuv]: https://github.com/orgs/nodejs/teams/libuv
[@nodejs/linting]: https://github.com/orgs/nodejs/teams/linting
[@nodejs/node-api]: https://github.com/orgs/nodejs/teams/node-api
[@nodejs/npm]: https://github.com/orgs/nodejs/teams/npm
[@nodejs/performance]: https://github.com/orgs/nodejs/teams/performance
[@nodejs/post-mortem]: https://github.com/orgs/nodejs/teams/post-mortem
[@nodejs/process]: https://github.com/orgs/nodejs/teams/process
[@nodejs/python]: https://github.com/orgs/nodejs/teams/python
[@nodejs/repl]: https://github.com/orgs/nodejs/teams/repl
[@nodejs/sqlite]: https://github.com/orgs/nodejs/teams/sqlite
[@nodejs/streams]: https://github.com/orgs/nodejs/teams/streams
[@nodejs/test_runner]: https://github.com/orgs/nodejs/teams/test_runner
[@nodejs/testing]: https://github.com/orgs/nodejs/teams/testing
[@nodejs/timers]: https://github.com/orgs/nodejs/teams/timers
[@nodejs/tsc]: https://github.com/orgs/nodejs/teams/tsc
[@nodejs/url]: https://github.com/orgs/nodejs/teams/url
[@nodejs/v8-inspector]: https://github.com/orgs/nodejs/teams/v8-inspector
[@nodejs/zlib]: https://github.com/orgs/nodejs/teams/zlib
[Deprecation]: https://en.wikipedia.org/wiki/Deprecation
[SECURITY.md]: https://github.com/nodejs/node/blob/HEAD/SECURITY.md
[Stability Index]: ../api/documentation.md#stability-index
[TSC]: https://github.com/nodejs/TSC
[`--pending-deprecation`]: ../api/cli.md#--pending-deprecation
[`--throw-deprecation`]: ../api/cli.md#--throw-deprecation
[`@node-core/utils`]: https://github.com/nodejs/node-core-utils
[aix]: https://github.com/orgs/nodejs/teams/platform-aix
[arm]: https://github.com/orgs/nodejs/teams/platform-arm
[backporting guide]: backporting-to-release-lines.md
[commit message guidelines]: pull-requests.md#commit-message-guidelines
[commit-example]: https://github.com/nodejs/node/commit/b636ba8186
[commit-queue.md]: ./commit-queue.md
[freebsd]: https://github.com/orgs/nodejs/teams/platform-freebsd
[git-email]: https://help.github.com/articles/setting-your-commit-email-address-in-git/
[git-node]: https://github.com/nodejs/node-core-utils/blob/HEAD/docs/git-node.md
[git-node-metadata]: https://github.com/nodejs/node-core-utils/blob/HEAD/docs/git-node.md#git-node-metadata
[git-username]: https://help.github.com/articles/setting-your-username-in-git/
[large pull requests]: large-pull-requests.md
[macos]: https://github.com/orgs/nodejs/teams/platform-macos
[node-core-utils-credentials]: https://github.com/nodejs/node-core-utils#setting-up-credentials
[node-core-utils-issues]: https://github.com/nodejs/node-core-utils/issues
[ppc]: https://github.com/orgs/nodejs/teams/platform-ppc
[s390]: https://github.com/orgs/nodejs/teams/platform-s390
[semantic versioning]: https://semver.org/
[smartos]: https://github.com/orgs/nodejs/teams/platform-smartos
[unreliable tests]: https://github.com/nodejs/node/issues?q=is%3Aopen+is%3Aissue+label%3A%22CI+%2F+flaky+test%22
[windows]: https://github.com/orgs/nodejs/teams/platform-windows
[windows-arm]: https://github.com/orgs/nodejs/teams/platform-windows-arm