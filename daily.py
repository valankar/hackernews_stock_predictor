#!/usr/bin/env python3
"""Run daily functions."""

from retrying import retry

import hackernews
import plot
import predict
import stocks


@retry(wait_exponential_multiplier=5000, wait_exponential_max=50000, stop_max_delay=300000)
def retry_hackernews():
    """Get hackernews data with retries."""
    hackernews.main()


def main():
    """Main."""
    retry_hackernews()
    stocks.main()
    predict.main()
    plot.main()


if __name__ == '__main__':
    main()
