---
title: Production Incident Runbook
access: engineering
updated: 2026-04-18
---
# Production Incident Runbook

## Severity levels
Sev1 means a full outage or data loss and pages the on-call engineer immediately. Sev2 means a major feature is degraded. Sev3 is a minor issue handled during business hours.

## First 15 minutes
Acknowledge the page, open an incident channel, assign an incident commander, and post a status update every 30 minutes. Roll back the latest deploy first if the incident began after a release.

## After the incident
Write a blameless postmortem within 5 business days that lists the timeline, root cause, and action items with owners.
