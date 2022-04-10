# snc4onnx
Simple tool to combine onnx models. **S**imple **N**etwork **C**ombine Tool for **ONNX**.


## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& pip install -U onnx-simplifier \
&& pip install -U snc4onnx
```
### 1-2. Docker
```bash
### docker pull
$ docker pull pinto0309/snc4onnx:latest

### docker build
$ docker build -t pinto0309/snc4onnx:latest .

### docker run
$ docker run --rm -it -v `pwd`:/workdir pinto0309/snc4onnx:latest
$ cd /workdir
```

## 2. CLI Usage
```bash
$ snc4onnx -h

usage:
  snc4onnx [-h]
    --input_onnx_file_paths INPUT_ONNX_FILE_PATHS [INPUT_ONNX_FILE_PATHS ...]
    --op_prefixes_after_merging OP_PREFIXES_AFTER_MERGING [OP_PREFIXES_AFTER_MERGING ...]
    --srcop_destop SRCOP_DESTOP [SRCOP_DESTOP ...]
    [--output_onnx_file_path OUTPUT_ONNX_FILE_PATH]
    [--output_of_onnx_file_in_the_process_of_fusion]
    [--non_verbose]

optional arguments:
  -h, --help
        show this help message and exit
  --input_onnx_file_paths INPUT_ONNX_FILE_PATHS [INPUT_ONNX_FILE_PATHS ...]
        Input onnx file paths. At least two onnx files must be specified.
  --op_prefixes_after_merging OP_PREFIXES_AFTER_MERGING [OP_PREFIXES_AFTER_MERGING ...]
        Since a single ONNX file cannot contain multiple OPs with the same name,
        a prefix is added to all OPs in each input ONNX model to avoid duplication.
        Specify the same number of paths as input_onnx_file_paths.
        e.g. --op_prefixes_after_merging model1_prefix model2_prefix model3_prefix ...
  --srcop_destop SRCOP_DESTOP [SRCOP_DESTOP ...]
        The names of the output OP to join from and the input OP to join to are
        out1 in1 out2 in2 out3 in3 .... format.
        In other words, to combine model1 and model2,
        --srcop_destop model1_out1 model2_in1 model1_out2 model2_in2
        Also, --srcop_destop can be specified multiple times.
        The first --srcop_destop specifies the correspondence between model1 and model2,
        and the second --srcop_destop specifies the correspondence
        between model1 and model2 combined and model3.
        It is necessary to take into account that the prefix specified
        in op_prefixes_after_merging is given at the beginning of each OP name.
        e.g. To combine model1 with model2 and model3.
        --srcop_destop model1_src_op1 model2_dest_op1 model1_src_op2 model2_dest_op2 ...
        --srcop_destop comb_model12_src_op1 model3_dest_op1 comb_model12_src_op2 model3_dest_op2 ...
  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        Output onnx file path.
  --output_of_onnx_file_in_the_process_of_fusion
        Output of onnx files in the process of fusion.
  --non_verbose
        Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
$ python
>>> from snc4onnx import combine
>>> help(combine)

Help on function combine in module snc4onnx.onnx_network_combine:

combine(
    input_onnx_file_paths: List[str],
    op_prefixes_after_merging: List[str],
    srcop_destop: List[str],
    output_onnx_file_path: Union[str, NoneType] = 'merged_model.onnx',
    output_of_onnx_file_in_the_process_of_fusion: Union[bool, NoneType] = False,
    non_verbose: Union[bool, NoneType] = False
)

    Parameters
    ----------
    input_onnx_file_paths: List[str]
        Input onnx file paths. At least two onnx files must be specified.
        e.g. input_onnx_file_paths=["model1.onnx","model2.onnx","model3.onnx", ...]

    op_prefixes_after_merging: List[str]
        Since a single ONNX file cannot contain multiple OPs with the same name,
        a prefix is added to all OPs in each input ONNX model to avoid duplication.
        Specify the same number of paths as input_onnx_file_paths.
        e.g. op_prefixes_after_merging = ["model1_prefix","model2_prefix","model3_prefix", ...]

    srcop_destop: List[str]
        The names of the output OP to join from and the input OP to join to are
        [["out1,"in1"], ["out2","in2"], ["out3","in3"]] format.

        In other words, to combine model1 and model2,
        srcop_destop =
            [
                ["model1_out1_opname","model2_in1_opname"],
                ["model1_out2_opname","model2_in2_opname"]
            ]

        The first srcop_destop specifies the correspondence between model1 and model2,
        and the second srcop_destop specifies the correspondence
        between model1 and model2 combined and model3.
        It is necessary to take into account that the prefix specified
        in op_prefixes_after_merging is given at the beginning of each OP name.

        e.g. To combine model1 with model2 and model3.
        srcop_destop =
            [
                [
                    ["model1_src_op1","model2_dest_op1"],
                    ["model1_src_op2","model2_dest_op2"]
                ],
                [
                    ["combined_model1.2_src_op1","model3_dest_op1"],
                    ["combined_model1.2_src_op2","model3_dest_op2"]
                ],
                ...
            ]

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        Default: 'merged_model.onnx'

    output_of_onnx_file_in_the_process_of_fusion: Optional[bool]
        Output of onnx files in the process of fusion.
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False
```