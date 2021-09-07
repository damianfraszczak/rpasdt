# Development setup and requirements

This document contains instructions to help you contribute to this project.

## Table of Content
- [Local development setup](#-local-development-setup)
- [Release a version](#-releasing-a-new-version)
- [Pre-commit hook](#precommit-hooks)


## Local development setup

We use Docker to setup the necessary environment and tools to build this
project. Go to the [docker's README](docs/files/docker.md) page to get instructions.

You will have to manually install a database engine and create a database yourself.


### Release a version

- Merge your PR into **`master`**
- Update changelog in api_clients/CHANGELOG.md
- Change the version in api_clients/version.py
- Commit. `git commit -m 'Release version x.y.z'`
- Tag the commit. `git tag -a x.y.z -m 'Release version x.y.z'`
- Push (do not forget --tags). `git push origin master --tags`
- Merge master into release when tests passes in Travis.
- Push to **`release`** branch.

When tests passes in Travis in release branch it automatically builds the package and
publishes it to our PyPi server.


## Pre-commit Hooks

This project supports [**pre-commit**](https://pre-commit.com/). To use it please install it
in the `pip install pre-commit` and then run `pre-commit install` and you are ready to go.
Bunch of checks will be executed before commit and files will be formatted correctly.
