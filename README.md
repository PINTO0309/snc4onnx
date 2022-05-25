# snc4onnx
Simple tool to combine(merge) onnx models. **S**imple **N**etwork **C**ombine Tool for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/snc4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/snc4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/snc4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/snc4onnx?color=2BAF2B)](https://pypi.org/project/snc4onnx/) [![CodeQL](https://github.com/PINTO0309/snc4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/snc4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/170151148-1b33c37f-9a97-4c55-a6ae-f4abccdcfc28.png" />
</p>

## 1. Setup

### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& pip install -U onnx-simplifier \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U snc4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```bash
$ snc4onnx -h

usage:
  snc4onnx [-h]
    --input_onnx_file_paths INPUT_ONNX_FILE_PATHS [INPUT_ONNX_FILE_PATHS ...]
    --srcop_destop SRCOP_DESTOP [SRCOP_DESTOP ...]
    [--op_prefixes_after_merging OP_PREFIXES_AFTER_MERGING [OP_PREFIXES_AFTER_MERGING ...]]
    [--output_onnx_file_path OUTPUT_ONNX_FILE_PATH]
    [--output_of_onnx_file_in_the_process_of_fusion]
    [--non_verbose]

optional arguments:
  -h, --help
    show this help message and exit

  --input_onnx_file_paths INPUT_ONNX_FILE_PATHS [INPUT_ONNX_FILE_PATHS ...]
    Input onnx file paths. At least two onnx files must be specified.

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

  --op_prefixes_after_merging OP_PREFIXES_AFTER_MERGING [OP_PREFIXES_AFTER_MERGING ...]
    Since a single ONNX file cannot contain multiple OPs with the same name,
    a prefix is added to all OPs in each input ONNX model to avoid duplication.
    Specify the same number of paths as input_onnx_file_paths.
    e.g. --op_prefixes_after_merging model1_prefix model2_prefix model3_prefix ...

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
  srcop_destop: List[str],
  op_prefixes_after_merging: Union[List[str], NoneType] = [],
  input_onnx_file_paths: Union[List[str], NoneType] = [],
  onnx_graphs: Union[List[onnx.onnx_ml_pb2.ModelProto], NoneType] = [],
  output_onnx_file_path: Union[str, NoneType] = '',
  output_of_onnx_file_in_the_process_of_fusion: Union[bool, NoneType] = False,
  non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    srcop_destop: List[str]
        The names of the output OP to join from and the input OP to join to are
        [["out1","in1"], ["out2","in2"], ["out3","in3"]] format.

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

    op_prefixes_after_merging: List[str]
        Since a single ONNX file cannot contain multiple OPs with the same name,
        a prefix is added to all OPs in each input ONNX model to avoid duplication.
        Specify the same number of paths as input_onnx_file_paths.
        e.g. op_prefixes_after_merging = ["model1_prefix","model2_prefix","model3_prefix", ...]

    input_onnx_file_paths: Optional[List[str]]
        Input onnx file paths. At least two onnx files must be specified.
        Either input_onnx_file_paths or onnx_graphs must be specified.
        onnx_graphs If specified, ignore input_onnx_file_paths and process onnx_graphs.
        e.g. input_onnx_file_paths = ["model1.onnx", "model2.onnx", "model3.onnx", ...]

    onnx_graphs: Optional[List[onnx.ModelProto]]
        List of onnx.ModelProto. At least two onnx graphs must be specified.
        Either input_onnx_file_paths or onnx_graphs must be specified.
        onnx_graphs If specified, ignore input_onnx_file_paths and process onnx_graphs.
        e.g. onnx_graphs = [graph1, graph2, graph3, ...]

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If not specified, .onnx is not output.
        Default: ''

    output_of_onnx_file_in_the_process_of_fusion: Optional[bool]
        Output of onnx files in the process of fusion.
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    combined_graph: onnx.ModelProto
        Combined onnx ModelProto
```

## 4. CLI Execution
```bash
$ snc4onnx \
--input_onnx_file_paths crestereo_init_iter2_120x160.onnx crestereo_next_iter2_240x320.onnx \
--srcop_destop output flow_init \
--op_prefixes_after_merging init next
```

## 5. In-script Execution
### 5-1. ONNX files
```python
from snc4onnx import combine

combined_graph = combine(
    srcop_destop = [
        ['output', 'flow_init']
    ],
    op_prefixes_after_merging = [
        'init',
        'next',
    ],
    input_onnx_file_paths = [
        'crestereo_init_iter2_120x160.onnx',
        'crestereo_next_iter2_240x320.onnx',
    ],
    non_verbose = True,
)
```
### 5-2. onnx.ModelProtos
```python
from snc4onnx import combine

combined_graph = combine(
    srcop_destop = [
        ['output', 'flow_init']
    ],
    op_prefixes_after_merging = [
        'init',
        'next',
    ],
    onnx_graphs = [
        graph1,
        graph2,
        graph3,
    ],
    non_verbose = True,
)
```

## 6. Sample
### 6-1 INPUT <-> OUTPUT
- Summary

  ![image](https://user-images.githubusercontent.com/33194443/162609071-ddd7ba38-ad05-4a15-ad13-9ddfe2adec99.png)

- Model.1

  ![image](https://user-images.githubusercontent.com/33194443/162609250-e5a7f915-52f9-4550-8d1f-bcc02a75ff90.png)

- Model.2

  ![image](https://user-images.githubusercontent.com/33194443/162609270-7df7579f-2ba2-4ddd-abc7-4fef997fab44.png)

- Merge

  ```bash
  $ snc4onnx \
  --input_onnx_file_paths crestereo_init_iter2_120x160.onnx crestereo_next_iter2_240x320.onnx \
  --op_prefixes_after_merging init next \
  --srcop_destop output flow_init
  ```

- Result

  ![image](https://user-images.githubusercontent.com/33194443/162609353-6e50c33c-ff0d-4cca-93fb-98636b605dbe.png)
  ![image](https://user-images.githubusercontent.com/33194443/162609415-cb302fee-90f4-41a7-aadf-08d6de29b40c.png)

### 6-2 INPUT + INPUT
- Summary

  ![image](https://user-images.githubusercontent.com/33194443/166130725-4fdbb466-08ad-4ab3-9b24-f3b93819d36d.png)

- Model.1

  ![image](https://user-images.githubusercontent.com/33194443/166130641-ff11c55b-f7e1-4231-b410-d94afa91418d.png)

- Model.2

  ![image](https://user-images.githubusercontent.com/33194443/166130699-fff17184-8586-4c86-a9b5-64f9566572fa.png)

- Merge

  ```bash
  $ snc4onnx \
  --input_onnx_file_paths objectron_camera_224x224.onnx objectron_chair_224x224.onnx \
  --srcop_destop input_1 input_1 \
  --op_prefixes_after_merging camera chair \
  --output_onnx_file_path objectron_camera_chair_224x224.onnx
  ```

- Result

  ![image](https://user-images.githubusercontent.com/33194443/166130549-d46f48b1-0b8b-40ad-bc9d-16b2046c963f.png)
  ![image](https://user-images.githubusercontent.com/33194443/166130582-8abaefbc-bcb5-4b3d-9b21-3da1b4c3460b.png)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md
2. https://github.com/PINTO0309/sne4onnx
3. https://github.com/PINTO0309/snd4onnx
4. https://github.com/PINTO0309/scs4onnx
5. https://github.com/PINTO0309/sog4onnx
6. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
