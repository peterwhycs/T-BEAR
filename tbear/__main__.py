#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ("main",)

import sys


def main(args=None):
    """Main routine of T-BEAR."""
    if args is None:
        args = sys.argv[1:]


if __name__ == "__main__":
    main()
