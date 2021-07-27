#!/bin/bash

source ~/.bashrc
rpasdt
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

  package)
    exec pyinstaller -n rpasdt /app/rpasdt/main.py
    ;;
  *)
    echo "
    Usage: docker-compose run rpasdt <command>

    Commands:
        help: this help text

        shell: Open an interractive shell in the container

        run: Run the application

        package: Generate an executable artifact
    "
    ;;

esac
