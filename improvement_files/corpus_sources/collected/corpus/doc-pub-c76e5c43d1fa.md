## Disclosure policy

Here is the security disclosure policy for Node.js

* The security report is received and is assigned a primary handler. This
  person will coordinate the fix and release process. The problem is validated
  against all supported Node.js versions. Once confirmed, a list of all affected
  versions is determined. Code is audited to find any potential similar
  problems. Fixes are prepared for all supported releases.
  These fixes are not committed to the public repository but rather held locally
  pending the announcement.

* A suggested embargo date for this vulnerability is chosen and a CVE (Common
  Vulnerabilities and Exposures (CVE®)) is requested for the vulnerability.

* On the embargo date, a copy of the announcement is sent to the Node.js
  security mailing list. The changes are pushed to the public repository and new
  builds are deployed to nodejs.org. Within 6 hours of the mailing list being
  notified, a copy of the advisory will be published on the Node.js blog.

* Typically, the embargo date will be set 72 hours from the time the CVE is
  issued. However, this may vary depending on the severity of the bug or
  difficulty in applying a fix.

* This process can take some time, especially when we need to coordinate with
  maintainers of other projects. We will try to handle the bug as quickly as
  possible; however, we must follow the release process above to ensure that we
  handle disclosure consistently.