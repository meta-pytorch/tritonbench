---
name: fix-linting-errors
description: Fix linting errors in a PR or Diff.
---

# Fix linting errors

## Description

This skill fixes linting errors in a PR or Diff.

## Step 1: Check if ufmt is installed

Check if ufmt is installed by running `ufmt`.

If ufmt is not installed, install it by running `uv pip install -r requirements-fmt.txt`.

If in open source environment, find `requirements-fmt.txt` in the root of the repository.

If in Meta environment, find `requirements-fmt.txt` in `fbsource/tools/lint/pyfmt/reqs/requirements-fmt.txt`.

## Step 2: Run ufmt linter

Run `ufmt format <changed_file>` to format all the changed file in the Diff or PR.

