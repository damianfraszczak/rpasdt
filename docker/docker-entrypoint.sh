#!/bin/bash

source ~/.bashrc
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Docker service command
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

case "$1" in
  shell)
    exec /bin/bash
    ;;

  run)
    exec python /app/rpasdt/gui/main.py
    ;;

  cli)
    exec python /app/rpasdt/cli/main.py
    ;;

  python)
    exec python "${@:2}"
    ;;

  package)
    echo -e "\nBuilding an executable artifact."
    cd pyinstaller_config
    exec pyinstaller --onefile --windowed rpasdt.spec
    ;;

  makedoc)
    cd docs
    exec make html
    ;;

  *)

    echo "
    Usage: docker-compose run rpasdt <command>

    Commands:
        help: this help text

        shell: Open an interactive shell in the container

        run: Run the application GUI

        cli: Run CLI commands

        python: Run python shell

        package: Generate an executable artifact

        makedoc: Generate documentation
    "
    ;;

esac
