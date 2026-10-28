"""Intentionally corrupt the generated golden vector for the negative-control build."""
import argparse
from pathlib import Path


def corrupt(header):
    text = header.read_text()
    old = 'const float kExpected[] = {-2.000000000f,'
    if text.count(old) != 1:
        raise ValueError('Expected pristine ADD golden vector, refusing to patch')
    header.write_text(text.replace(old, 'const float kExpected[] = {99.000000000f,'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header', type=Path, required=True)
    corrupt(parser.parse_args().header)
