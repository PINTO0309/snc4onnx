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
        nargs='+',
        help='Input onnx file paths. At least two onnx files must be specified.'
    )
    parser.add_argument(
        '--op_prefix_after_merging',
        type=str,
        required=True,
        nargs='+',
        help='\
            Since a single ONNX file cannot contain multiple OPs with the same name, '+
            'a prefix is added to all OPs in each input ONNX model to avoid duplication. \n'+
            'Specify the same number of paths as input_onnx_file_paths. \n'+
            'e.g. --op_prefix_after_merging model1_prefix model2_prefix model3_prefix ...'
    )
    parser.add_argument(
        '--srcop_distop',
        type=str,
        required=True,
        nargs='+',
        action='append',
        help='\
            The names of the output OP to join from and the input OP to join to are '+
            'out1 in1 out2 in2 out3 in3 .... format. \n'+
            'In other words, to combine model1 and model2, '+
            '--srcop_distop model1_out1 model2_in1 model1_out2 model2_in2 \n'+
            'Also, --srcop_distop can be specified multiple times. \n'+
            'The first --srcop_distop specifies the correspondence between model1 and model2, '+
            'and the second --srcop_distop specifies the correspondence between model1 and model2 combined and model3. \n'+
            'It is necessary to take into account that the prefix specified '+
            'in op_prefix_after_merging is given at the beginning of each OP name. \n'+
            'e.g. To combine model1 with model2 and model3. \n'+
            '--srcop_distop model1_src_op1 model2_dist_op1 model1_src_op2 model2_dist_op2 ... \n'+
            '--srcop_distop combined_model_src_op1 model3_dist_op1 combined_model_src_op2 model3_dist_op2 ...'
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

    # Two or more ONNX files must be specified
    if len(args.input_onnx_file_paths) <= 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Two or more input_onnx_file_paths must be specified.'
        )
        sys.exit(1)

    # Match check between number of onnx files and number of prefixes
    if len(args.input_onnx_file_paths) != len(args.op_prefix_after_merging):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The number of input_onnx_file_paths must match the number of op_prefix_after_merging.'
        )
        sys.exit(1)

    # Duplicate prefix check
    def has_duplicates(seq):
        seen = []
        unique_list = [x for x in seq if x not in seen and not seen.append(x)]
        return len(seq) != len(unique_list)

    if has_duplicates(args.op_prefix_after_merging):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Duplicate values cannot be specified for op_prefix_after_merging.'
        )
        sys.exit(1)


    # onnx x2 -> srcop_distop x1
    # onnx x3 -> srcop_distop x2
    # onnx x4 -> srcop_distop x3
    if len(args.input_onnx_file_paths) - 1 != len(args.srcop_distop):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The number of srcop_distops must be (number of input_onnx_file_paths - 1).'
        )
        sys.exit(1)

    # MODEL_INDX print
    for idx, input_onnx_file_path in enumerate(args.input_onnx_file_paths):
        if not args.non_verbose:
            print(f'{Color.GREEN}INFO:{Color.RESET} MODEL_INDX={idx}: {input_onnx_file_path}')


    print(args.srcop_distop)
    print(len(args.srcop_distop))

    # Model combine

if __name__ == '__main__':
    main()