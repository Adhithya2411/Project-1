# Security Incident Response Runbook

Owner: Security Engineering · Classification: internal

## 1. Severity definitions

- **SEV-1** — confirmed unauthorised access to production data, or an active
  compromise of a production credential. Page immediately.
- **SEV-2** — credible indication of compromise without confirmed data access;
  or loss of a security control (for example, audit logging stopped).
- **SEV-3** — a vulnerability with no evidence of exploitation.

## 2. First thirty minutes

1. Declare the incident in `#security-incidents` and open a bridge.
2. Assign an incident commander. The commander does not perform remediation.
3. Preserve evidence before remediating: snapshot affected hosts, export the
   relevant audit logs, and record the time of each action.
4. Do not rotate credentials until evidence preservation is complete, unless an
   active compromise is ongoing.

## 3. Error code SEC-3301 — audit log gap detected

The audit pipeline detected a gap in the immutable audit stream. Treat as SEV-2
until the gap is explained.

Remediation for SEC-3301:

1. Confirm the gap window from the audit ingestion dashboard.
2. Check whether a deployment or a log rotation coincides with the window.
3. If unexplained, escalate to the Security Engineering lead and treat all
   activity in the gap window as unverified.

## 4. Access review

Access to restricted document classes is reviewed quarterly by the owning
function. Finance-classified material is reviewed by the Finance Director.
Revocations take effect at the next directory sync, which runs hourly.

## 5. Post-incident

A written post-incident review is required for all SEV-1 and SEV-2 incidents
within ten working days. The review records timeline, contributing factors, and
actions with named owners and dates. Blameless language is required.
