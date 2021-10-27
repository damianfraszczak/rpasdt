"""Entrypoint for CLI operations."""
import sys

import fire

from rpasdt.cli import CLIInterface
from rpasdt.common.exceptions import log_error


def main():
    sys.excepthook = log_error
    fire.Fire(CLIInterface)


if __name__ == "__main__":
    main()
