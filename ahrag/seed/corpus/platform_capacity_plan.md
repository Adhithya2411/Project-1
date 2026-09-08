# Platform Engineering Capacity Plan — H2 2026

Owner: Head of Platform Engineering · Status: approved

## 1. Rota expansion

The payments on-call rota expands from 5 to 7 engineers effective 1 July 2026.
Two engineers transfer from the infrastructure rota. This addresses risk R-03
recorded in the Project Atlas Q2 status report.

The expansion reduces the on-call frequency from one week in five to one week in
seven, and brings the rota into line with the internal target of no more than
one primary on-call week per engineer per six weeks.

## 2. Service ownership

| Service | Primary owner | Secondary |
|---|---|---|
| payments-api | Payments squad | Platform core |
| checkout-web | Checkout squad | Payments squad |
| ledger-worker | Payments squad | Data platform |

## 3. Capacity allocation H2 2026

- 45% Atlas migration completion, including reconciliation parity work
- 25% reliability and error-budget work, principally ERR-5041 mitigation
- 20% platform upgrades (Postgres 16, Kubernetes 1.31)
- 10% unplanned / incident response

## 4. Constraints

The plan assumes no additional hiring beyond the 17 engineering roles already
approved for 2026. If the Atlas legacy decommission slips beyond September 2026,
the reliability allocation is reduced to 15% and the shortfall is absorbed by the
Atlas allocation.

## 5. Connection pool remediation

The connection pool sizing fix for `payments-api` is scheduled for the sprint
beginning 2026-07-13, and is expected to return the ERR-5041 rate to its 0.3%
baseline.
