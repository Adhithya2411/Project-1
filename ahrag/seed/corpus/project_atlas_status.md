# Project Atlas — Status Report, Q2 2026

Prepared by: Programme Management Office · Reporting period: 1 Apr – 30 Jun 2026

## 1. Summary

Project Atlas is the migration of the checkout stack from the legacy monolith to
the `payments-api` service. The project is **amber**. The core migration is on
track, but the reconciliation workstream has slipped by four weeks.

## 2. Milestones

| Milestone | Planned | Forecast | Status |
|---|---|---|---|
| Dual-write to new ledger | 2026-04-15 | 2026-04-14 | Complete |
| 10% traffic cutover | 2026-05-01 | 2026-05-06 | Complete |
| 50% traffic cutover | 2026-06-01 | 2026-06-03 | Complete |
| Reconciliation parity | 2026-06-30 | 2026-07-28 | At risk |
| Legacy decommission | 2026-08-31 | 2026-09-30 | At risk |

## 3. Risks

**R-01 (High): Reconciliation parity slip.** The daily reconciliation job does
not yet match legacy output for partial refunds. Four weeks of engineering
effort remain. Owner: payments tech lead.

**R-02 (Medium): Acquirer timeout rate.** ERR-5041 occurrences rose from a
baseline of 0.3% to 0.9% of capture requests after the 50% cutover. The
platform team attributes this to connection pool sizing on the new service and
has a fix in test. Owner: Platform Engineering.

**R-03 (Medium): On-call load.** The payments on-call rota absorbed both the
legacy and new service during cutover. Rota size increases from 5 to 7 engineers
in July.

## 4. Dependencies

Atlas depends on the Finance close calendar for its reconciliation sign-off
window and on the Platform Engineering capacity plan for the July rota
expansion. Neither dependency is currently blocking.

## 5. Decisions requested

The steering group is asked to approve a four-week extension to the legacy
decommission date, moving it from 2026-08-31 to 2026-09-30.
