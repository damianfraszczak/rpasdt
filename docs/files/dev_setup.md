## Persistent volumes
If you want to have persistent volumes you can:

* use `docker-compose.dev.yml` file for Docker Compose, e.g. `docker-compose -f docker-compose.dev.yml run rpasdt gui`
* copy `docker-compose.dev.yml` and name it `docker-compose.override.yml`.
  It is gitignored so you won't add this to repo by accident.

# Pre-Commit

This project supports [pre-commit](https://pre-commit.com/). To use it please install it in the `pip install pre-commit` and then run `pre-commit install` and you are ready to go.

Best way to use pre-commit is to install it globaly.

Pre-commit works on staged files while commiting. To run it without a command one should run `pre-commit run`. Changes has to be staged.

To run pre-commit hooks on all changes in the branch:

1.  Sync branch with master
1.  Run `git diff --name-only --diff-filter=MA origin/main | xargs pre-commit run --files`

For branches that are not based on `master` you might replace `origin/main` with `origin/{your_branch}`

## Windows configuration

## Linux configuration
Sample `docker-compose.override.yml` file for Linux users:
```docker-compose
version: "3.3"
services:
  rpasdt:
    environment:
      - DISPLAY={YOUR_IP}:0.0
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ../src/rpasdt:/app/rpasdt
      - ../dist:/app/pyinstaller_config/dist
```
