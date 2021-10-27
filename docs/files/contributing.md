# Development setup and requirements

This document contains instructions to help you contribute to this project.

## Table of Content
- [Local development setup](#-local-development-setup)
- [Release a version](#-releasing-a-new-version)
- [Pre-commit hook](#precommit-hooks)

## Local development setup

We use Docker to setup the necessary environment and tools to build this
project. Go to the [docker's README](docker.md) page to get instructions.

### Release a version

- Merge your PR into **`master`**
- Update changelog in CHANGELOG.md
- Change the version in src/rpasdt/version.py
- Commit. `git commit -m 'Release version x.y.z'`
- Tag the commit. `git tag -a x.y.z -m 'Release version x.y.z'`
- Push (do not forget --tags). `git push origin master --tags`



## Pre-commit Hooks

This project supports [**pre-commit**](https://pre-commit.com/). To use it please install it
in the `pip install pre-commit` and then run `pre-commit install` and you are ready to go.
Bunch of checks will be executed before commit and files will be formatted correctly.

Pre-commit works on staged files while commiting. To run it without a command one should run `pre-commit run`. Changes has to be staged.

To run pre-commit hooks on all changes in the branch:

1.  Sync branch with main
1.  Run `git diff --name-only --diff-filter=MA origin/main | xargs pre-commit run --files`

For branches that are not based on `main` you might replace `origin/main` with `origin/{your_branch}`
