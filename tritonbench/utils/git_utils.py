import os
import subprocess
from datetime import datetime


def get_branch(repo: str, commit: str) -> str:
    """Get branch name with a given commit in a git repo.
    repo: local git repo path
    commit: hash of a commit
    If a commit does not belong to any branch, return "unknown"
    If a commit belongs to many branches, return the very first branch.
    """
    if repo == "unknown":
        return "unknown"
    assert os.path.exists(repo), f"{repo} path does not exist."
    cmd = ["git", "branch", "-a", "--contains", commit, "--no-color"]
    branch_names = subprocess.check_output(cmd, cwd=repo).decode().strip().splitlines()
    if not len(branch_names):
        return "unknown"
    if branch_names[0].startswith("* "):
        return branch_names[0][2:]
    return branch_names[0]


def get_commit_time(repo: str, commit: str) -> str:
    """Get commit time in a git repo.
    repo: URL or local git repo path
    commit: hash of a commit
    If a commit does not exist, return "unknown"
    """
    if repo == "unknown":
        return "unknown"
    assert os.path.exists(repo), f"{repo} path does not exist."
    git_date_cmd = ["git", "show", "--no-patch", "--format=%ci", commit]
    git_date = subprocess.check_output(git_date_cmd, cwd=repo).decode().strip()
    if not git_date:
        return "unknown"
    date_format = "%Y-%m-%d %H:%M:%S %z"
    parsed_date = datetime.strptime(git_date, date_format)
    return datetime.strftime(parsed_date, "%Y%m%d%H%M%S")


def get_current_hash(repo: str) -> str:
    """Get the HEAD hash of a git repo.
    repo: local git repo path"""
    if repo == "unknown":
        return "unknown"
    cmd = ["git", "rev-parse", "--verify", "HEAD"]
    output = subprocess.check_output(cmd, cwd=repo).decode().strip()
    return output
