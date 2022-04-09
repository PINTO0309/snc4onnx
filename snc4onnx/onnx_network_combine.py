#! /usr/bin/env python

import sys
import os
from argparse import ArgumentParser
import onnx
from typing import List

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def combine():
    pass

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_paths',
        type=str,
        required=True,
        nargs='*',
        help='Input onnx file paths.'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    """
    Pattern
        [1] out x1 : in x1
        [2] out x1 : in xN
    """

    for idx, input_onnx_file_path in enumerate(args.input_onnx_file_paths):
        # file existence check
        if not os.path.exists(input_onnx_file_path) or \
            not os.path.isfile(input_onnx_file_path) or \
            not os.path.splitext(input_onnx_file_path)[-1] == '.onnx':
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'The specified file (.onnx) does not exist. or not an onnx file. File: {input_onnx_file_path}'
            )
            sys.exit(1)

        # MODEL_INDX print
        if not args.non_verbose:
            print(f'{Color.GREEN}INFO:{Color.RESET} MODEL_INDX={idx}: {input_onnx_file_path}')

    # Model combine

if __name__ == '__main__':
    main()