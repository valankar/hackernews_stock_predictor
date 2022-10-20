#!/usr/bin/env python3
"""Run daily functions."""

import hackernews
import plot
import predict
import stocks


def main():
    """Main."""
    hackernews.main()
    stocks.main()
    predict.main()
    plot.main()


if __name__ == '__main__':
     main()
