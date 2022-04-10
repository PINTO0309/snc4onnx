import onnx
from onnxsim import simplify

INIT_MODEL = [
    'crestereo_init_iter2_120x160.onnx',
]

NEXT_MODEL = [
    'crestereo_next_iter2_240x320.onnx',
]

for init_model, next_model in zip(INIT_MODEL, NEXT_MODEL):
    model1 = onnx.load(init_model)
    model1 = onnx.compose.add_prefix(model1, prefix='init_')

    model2 = onnx.load(next_model)
    model2 = onnx.compose.add_prefix(model2, prefix='next_')
    combined_model = onnx.compose.merge_models(
        model1, model2,
        io_map=[('init_output', 'next_flow_init')]
    )
    file_name = \
        next_model.split('_')[0] + '_' + \
        'combined_' + \
        next_model.split('_')[2] + '_' + \
        next_model.split('_')[3].split('.')[0] + \
        '.onnx'
    onnx.save(combined_model, file_name)
