#! /usr/bin/env python

import sys
import os
import traceback
from argparse import ArgumentParser
import onnx
from onnxsim import simplify
from typing import Optional, List

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


def combine(
    input_onnx_file_paths: List[str],
    op_prefixes_after_merging: List[str],
    srcop_destop: List[str],
    output_onnx_file_path: Optional[str] = 'merged_model.onnx',
    output_of_onnx_file_in_the_process_of_fusion: Optional[bool] = False,
    non_verbose: Optional[bool] = False,
):
    """
    Parameters
    ----------
    input_onnx_file_paths: List[str]
        Input onnx file paths. At least two onnx files must be specified.
        e.g. input_onnx_file_paths=["model1.onnx","model2.onnx","model3.onnx", ...]

    op_prefixes_after_merging: List[str]
        Since a single ONNX file cannot contain multiple OPs with the same name,\n\
        a prefix is added to all OPs in each input ONNX model to avoid duplication.\n\
        Specify the same number of paths as input_onnx_file_paths.\n\
        e.g. op_prefixes_after_merging = ["model1_prefix","model2_prefix","model3_prefix", ...]

    srcop_destop: List[str]
        The names of the output OP to join from and the input OP to join to are\n\
        [["out1,"in1"], ["out2","in2"], ["out3","in3"]] format.\n\n\
        In other words, to combine model1 and model2,\n\
        srcop_destop = \n\
            [
                ["model1_out1_opname","model2_in1_opname"],\n\
                ["model1_out2_opname","model2_in2_opname"]\n\
            ]\n\n\
        The first srcop_destop specifies the correspondence between model1 and model2, and \n\
        the second srcop_destop specifies the correspondence between model1 and model2 combined and model3.\n\
        It is necessary to take into account that the prefix specified \n\
        in op_prefixes_after_merging is given at the beginning of each OP name.\n\n\
        e.g. To combine model1 with model2 and model3.\n\
        srcop_destop = \n\
            [\n\
                [\n\
                    ["model1_src_op1","model2_dest_op1"],\n\
                    ["model1_src_op2","model2_dest_op2"]\n\
                ],\n\
                [\n\
                    ["combined_model1.2_src_op1","model3_dest_op1"],\n\
                    ["combined_model1.2_src_op2","model3_dest_op2"]\n\
                ],\n\
                ...\n\
            ]

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        Default: 'merged_model.onnx'

    output_of_onnx_file_in_the_process_of_fusion: Optional[bool]
        Output of onnx files in the process of fusion.\n\
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False
    """

    for idx, input_onnx_file_path in enumerate(input_onnx_file_paths):
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
    if len(input_onnx_file_paths) <= 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Two or more input_onnx_file_paths must be specified.'
        )
        sys.exit(1)

    # Match check between number of onnx files and number of prefixes
    if len(input_onnx_file_paths) != len(op_prefixes_after_merging):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The number of input_onnx_file_paths must match the number of op_prefixes_after_merging.'
        )
        sys.exit(1)

    # Duplicate prefix check
    def has_duplicates(seq):
        seen = []
        unique_list = [x for x in seq if x not in seen and not seen.append(x)]
        return len(seq) != len(unique_list)

    if has_duplicates(op_prefixes_after_merging):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Duplicate values cannot be specified for op_prefixes_after_merging.'
        )
        sys.exit(1)

    # onnx x2 -> srcop_destop x1
    # onnx x3 -> srcop_destop x2
    # onnx x4 -> srcop_destop x3
    if len(input_onnx_file_paths) - 1 != len(srcop_destop):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The number of srcop_destops must be (number of input_onnx_file_paths - 1).'
        )
        sys.exit(1)

    # MODEL_INDX print
    for idx, (input_onnx_file_path, op_prefix_after_merging) in enumerate(zip(input_onnx_file_paths, op_prefixes_after_merging)):
        if not non_verbose:
            print(
                f'{Color.GREEN}INFO:{Color.RESET} '+
                f'MODEL_INDX={idx+1}: {input_onnx_file_path}, prefix="{op_prefix_after_merging}"'
            )

    # Combine
    ## 1. ONNX load
    onnx_graphs = []
    for onnx_path in input_onnx_file_paths:
        onnx_graphs.append(onnx.load(onnx_path))

    ## 2. Repeat Merge
    for model_idx in range(0, len(onnx_graphs) - 1):

        src_prefix = ''
        if model_idx == 0:
            src_model = onnx_graphs[model_idx]
            dest_model = onnx_graphs[model_idx+1]
            src_prefix = f'{op_prefixes_after_merging[model_idx]}_'
        else:
            src_model = combined_model
            dest_model = onnx_graphs[model_idx+1]
            src_prefix = ''

        dest_prefix = f'{op_prefixes_after_merging[model_idx+1]}_'

        src_model = onnx.compose.add_prefix(
            src_model,
            prefix=src_prefix
        )
        dest_model = onnx.compose.add_prefix(
            dest_model,
            prefix=dest_prefix
        )

        io_map = [(f'{src_prefix}{io_map_srcop_destop[0]}', f'{dest_prefix}{io_map_srcop_destop[1]}') for io_map_srcop_destop in srcop_destop]

        combined_model = onnx.compose.merge_models(
            src_model,
            dest_model,
            io_map=io_map
        )

        ## Output of onnx files in the process of fusion
        if output_of_onnx_file_in_the_process_of_fusion:
            temp_file_path = f'{os.path.splitext(output_onnx_file_path)[0]}_{model_idx+1}{os.path.splitext(output_onnx_file_path)[1]}'
            onnx.save(
                combined_model,
                temp_file_path
            )
            if not non_verbose:
                print(
                    f'{Color.GREEN}INFO:{Color.RESET} '+
                    f'Output the fusion result of model {model_idx+1} and model {model_idx+2}. File: {temp_file_path}'
                )

    ## 3. Optimize
    try:
        combined_model, check = simplify(combined_model)
    except Exception as e:
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'Failed to optimize the combined onnx file.'
            )
            tracetxt = traceback.format_exc().splitlines()[-1]
            print(f'{Color.YELLOW}WARNING:{Color.RESET} {tracetxt}')

    ## 4. Final save
    onnx.save(combined_model, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')


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
        '--op_prefixes_after_merging',
        type=str,
        required=True,
        nargs='+',
        help=\
            'Since a single ONNX file cannot contain multiple OPs with the same name, '+
            'a prefix is added to all OPs in each input ONNX model to avoid duplication. \n'+
            'Specify the same number of paths as input_onnx_file_paths. \n'+
            'e.g. --op_prefixes_after_merging model1_prefix model2_prefix model3_prefix ...'
    )
    parser.add_argument(
        '--srcop_destop',
        type=str,
        required=True,
        nargs='+',
        action='append',
        help=\
            'The names of the output OP to join from and the input OP to join to are '+
            'out1 in1 out2 in2 out3 in3 .... format. \n'+
            'In other words, to combine model1 and model2, '+
            '--srcop_destop model1_out1 model2_in1 model1_out2 model2_in2 \n'+
            'Also, --srcop_destop can be specified multiple times. \n'+
            'The first --srcop_destop specifies the correspondence between model1 and model2, '+
            'and the second --srcop_destop specifies the correspondence between model1 and model2 combined and model3. \n'+
            'It is necessary to take into account that the prefix specified '+
            'in op_prefixes_after_merging is given at the beginning of each OP name. \n'+
            'e.g. To combine model1 with model2 and model3. \n'+
            '--srcop_destop model1_src_op1 model2_dest_op1 model1_src_op2 model2_dest_op2 ... \n'+
            '--srcop_destop combined_model1.2_src_op1 model3_dest_op1 combined_model1.2_src_op2 model3_dest_op2 ...'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        default='merged_model.onnx',
        help='Output onnx file path.'
    )
    parser.add_argument(
        '--output_of_onnx_file_in_the_process_of_fusion',
        action='store_true',
        help='Output of onnx files in the process of fusion.'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    # Model combine
    print(len(args.srcop_destop))
    combine(
        input_onnx_file_paths=args.input_onnx_file_paths,
        op_prefixes_after_merging=args.op_prefixes_after_merging,
        srcop_destop=args.srcop_destop,
        output_onnx_file_path=args.output_onnx_file_path,
        output_of_onnx_file_in_the_process_of_fusion=args.output_of_onnx_file_in_the_process_of_fusion,
        non_verbose=args.non_verbose,
    )

if __name__ == '__main__':
    main()