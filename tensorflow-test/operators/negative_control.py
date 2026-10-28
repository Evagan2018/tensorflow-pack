import argparse
from pathlib import Path


def corrupt(header):
    text = header.read_text()
    if text.count('#define OPS_NEGATIVE_CONTROL 0') != 1:
        raise ValueError('Expected pristine negative-control switch')
    header.write_text(text.replace('#define OPS_NEGATIVE_CONTROL 0', '#define OPS_NEGATIVE_CONTROL 1'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enable deliberate ADD golden corruption only.')
    parser.add_argument('--header', type=Path, required=True)
    corrupt(parser.parse_args().header)
