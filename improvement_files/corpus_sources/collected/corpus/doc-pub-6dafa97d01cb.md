## Code of Conduct and Vulnerability Reporting Guidelines

When reporting security vulnerabilities, reporters must adhere to the following guidelines:

1. **Code of Conduct Compliance**: All security reports must comply with our
   [Code of Conduct](CODE_OF_CONDUCT.md). Reports that violate our code of conduct
   will not be considered and may result in being banned from future participation.

2. **No Harmful Actions**: Security research and vulnerability reporting must not:
   * Cause damage to running systems or production environments.
   * Disrupt Node.js development or infrastructure.
   * Affect other users' applications or systems.
   * Include actual exploits that could harm users.
   * Involve social engineering or phishing attempts.

3. **Responsible Testing**: When testing potential vulnerabilities:
   * Use isolated, controlled environments.
   * Do not test on production systems without prior authorization. Contact
     the Node.js Technical Steering Committee (<tsc@iojs.org>) for permission or open
     a HackerOne report.
   * Do not attempt to access or modify other users' data.
   * Immediately stop testing if unauthorized access is gained accidentally.

4. **Report Quality**
   * Provide clear, detailed steps to reproduce the vulnerability.
   * Include reproducible code written in JavaScript.
   * Include only the minimum proof of concept required to demonstrate the issue.
   * Remove any malicious payloads or components that could cause harm.

Failure to follow these guidelines may result in:

* Rejection of the vulnerability report.
* Forfeiture of any potential bug bounty.
* Temporary or permanent ban from the bug bounty program.
* Legal action in cases of malicious intent.