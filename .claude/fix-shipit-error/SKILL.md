# Fixing TritonBench ShipIt Error Task

This skill requires user to provide a ShipIt task number <TASK_NUMBER>.
Inspect ShipIt task, find out the most recent failed ShipIt Sandcastle Workflow link.
This will require human to approve access to the Sandcastle Workflow.

Follow the Workflow link, find out the related error log message that are helpful to fix this task.
Based on the error message, decide which case is the following:

1. Diff Train is not broken, but there are too many Diffs stuck in Diff Train that we need to land them sooner
2. Diff Train is broken and we need to fix Diff Train
3. There is one or more Diff landed but no pull request found, we need to create a PR to re-sync between internal and GitHub


## Case 1: Diff Train is not broken, but there are too many Diffs stuck in Diff Train that we need to land them

In Case 1, simply list all Diffs that needs to be landed.

## Case 2: Diff Train is broken and we need to fix Diff Train

In Case 2, find the failing Diff Train workflow link and read the logs.

## Case 3: There is one or more Diff landed but no pull request found, we need to create a PR

In case 3, do the following:

1. create a patch file that can be used to create a PR
2. add "@diff-train-skip-merge" to the summary section of the related Diffs
