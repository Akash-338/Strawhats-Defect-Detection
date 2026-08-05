import importlib, sys
print('python', sys.version)

# Check PyTorch
try:
    ts = importlib.util.find_spec('torch')
    print('torch_installed', bool(ts))
    if ts:
        import torch
        print('torch.cuda.is_available()', torch.cuda.is_available())
        print('torch.cuda.device_count()', torch.cuda.device_count())
        try:
            print('torch.cuda.get_device_name(0)', torch.cuda.get_device_name(0))
        except Exception as e:
            print('get_device_name_error', e)
except Exception as e:
    print('torch_check_error', e)

# Check TensorFlow
try:
    tf_spec = importlib.util.find_spec('tensorflow')
    print('tensorflow_installed', bool(tf_spec))
    if tf_spec:
        import tensorflow as tf
        print('tf.test.is_built_with_cuda()', tf.test.is_built_with_cuda())
        print('tf_gpus', tf.config.list_physical_devices('GPU'))
except Exception as e:
    print('tensorflow_check_error', e)
