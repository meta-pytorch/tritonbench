import subprocess


def checkout_submodules(repo_root):
    status = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"],
        cwd=repo_root,
        text=True,
    )
    missing_submodules = [
        line.split()[1] for line in status.splitlines() if line.startswith("-")
    ]
    if not missing_submodules:
        return

    cmd = ["git", "submodule", "update", "--init", "--recursive", *missing_submodules]
    subprocess.check_call(cmd, cwd=repo_root)
