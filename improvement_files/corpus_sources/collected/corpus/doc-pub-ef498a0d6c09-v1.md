# Node.js collaborator guide

## Contents

* [Issues and pull requests](#issues-and-pull-requests)
  * [Welcoming first-time contributors](#welcoming-first-time-contributors)
  * [Closing issues and pull requests](#closing-issues-and-pull-requests)
  * [Stale issues and pull requests](#stale-issues-and-pull-requests)
  * [Author ready pull requests](#author-ready-pull-requests)
  * [Handling own pull requests](#handling-own-pull-requests)
  * [Repository triage views](#repository-triage-views)
  * [Security issues](#managing-security-issues)
* [Accepting modifications](#accepting-modifications)
  * [Code reviews](#code-reviews)
  * [Consensus seeking](#consensus-seeking)
  * [Waiting for approvals](#waiting-for-approvals)
  * [Testing and CI](#testing-and-ci)
    * [Useful Jenkins CI jobs](#useful-jenkins-ci-jobs)
    * [Starting a Jenkins CI job](#starting-a-jenkins-ci-job)
  * [Internal vs. public API](#internal-vs-public-api)
  * [Breaking changes](#breaking-changes)
    * [Breaking changes and deprecations](#breaking-changes-and-deprecations)
    * [Breaking changes to internal elements](#breaking-changes-to-internal-elements)
    * [Unintended breaking changes](#unintended-breaking-changes)
      * [Reverting commits](#reverting-commits)
  * [Introducing new modules](#introducing-new-modules)
  * [Additions to Node-API](#additions-to-node-api)
  * [Deprecations](#deprecations)
  * [Involving the TSC](#involving-the-tsc)
* [Landing pull requests](#landing-pull-requests)
  * [Using the commit queue GitHub labels](#using-the-commit-queue-github-labels)
  * [Using `git-node`](#using-git-node)
  * [Technical HOWTO](#technical-howto)
  * [Troubleshooting](#troubleshooting)
  * [I made a mistake](#i-made-a-mistake)
  * [Long Term Support](#long-term-support)
    * [What is LTS?](#what-is-lts)
    * [How are LTS branches managed?](#how-are-lts-branches-managed)
    * [How can I help?](#how-can-i-help)
* [Who to CC in the issue tracker](#who-to-cc-in-the-issue-tracker)

This document explains how collaborators manage the Node.js project.
Collaborators should understand the
[guidelines for new contributors](../../CONTRIBUTING.md) and the
[project governance model](../../GOVERNANCE.md).