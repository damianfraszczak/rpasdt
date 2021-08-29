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
    exec su qtuser -c 'python3 /app/rpasdt/main.py'
    ;;

  python3)
    exec python3 "${@:2}"
    ;;

  package)
    exec pyinstaller --onefile --windowed /app/rpasdt/main.py
    ;;

  *)

    echo "
    Usage: docker-compose run rpasdt <command>

    Commands:
        help: this help text

        shell: Open an interactive shell in the container

        run: Run the application

        python3: Run python shell

        package: Generate an executable artifact
    "
    ;;

esac
