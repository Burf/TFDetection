import os
import tempfile

import tensorflow as tf

def tf2trt(model, save_path, dtype = tf.float32, dynamic_batch = True, memory_limit = 1, data = None):
    """
    dtype = dtype in [tf.int8, tf.float16, tf.float32, 'INT8', 'FP16', 'FP32']
    dynamic_batch = static batch size or dynamic batch size
    memory_limit = x.xx gigabyte
    data = [np.array, ...]
    """
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    
    if dtype not in [tf.int8, tf.float16, tf.float32, trt.TrtPrecisionMode.INT8, trt.TrtPrecisionMode.FP16, trt.TrtPrecisionMode.FP32]:
        raise ValueError("unknown dtype '{0}'".format(dtype))
        
    dtype_converter = {tf.int8:trt.TrtPrecisionMode.INT8, trt.TrtPrecisionMode.INT8:trt.TrtPrecisionMode.INT8,
                       tf.float16:trt.TrtPrecisionMode.FP16, trt.TrtPrecisionMode.FP16:trt.TrtPrecisionMode.FP16,
                       tf.float32:trt.TrtPrecisionMode.FP32, trt.TrtPrecisionMode.FP32:trt.TrtPrecisionMode.FP32}
    dtype = dtype_converter[dtype]
    
    with tempfile.TemporaryDirectory() as temp_path:
        if isinstance(model, tf.keras.Model):
            model.save(temp_path)
            model = temp_path
        
        trt_param = trt.TrtConversionParams(precision_mode = dtype if data is None else "INT8", 
                                            max_workspace_size_bytes = int(round((1 << 30) * memory_limit)),
                                            maximum_cached_engines = 1)
        converter = trt.TrtGraphConverterV2(input_saved_model_dir = model,
                                            conversion_params = trt_param,
                                            use_dynamic_shape = dynamic_batch, 
                                            dynamic_shape_profile_strategy = "Optimal")
        if data is not None:
            data = data if isinstance(data, list) or isinstance(data, tuple) else [data]
            def representative_dataset():
                for ds in zip(*data):
                    yield [np.expand_dims(d, axis = 0) for d in ds]
            converter.convert(calibration_input_fn = representative_dataset)
            converter.build(input_fn = representative_dataset)
        else:
            converter.convert()
        save_path = os.path.splitext(save_path)[0]
        os.makedirs(save_path, exist_ok = True)
        converter.save(save_path)
        return save_path

def load_trt(dir_path, predict = True):
    from tensorflow.python.saved_model import signature_constants, tag_constants
    
    model = tf.saved_model.load(dir_path, tags = [tag_constants.SERVING])
    if predict:
        signature_runner = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_keys = [tensor.name.split(":")[0] for tensor in signature_runner.inputs]
        def predict(*args, **kwargs):
            args = {k:v for k, v in zip(input_keys[:len(args)], args)}
            kwargs.update(args)
            pred = signature_runner(**{k:v if tf.is_tensor(v) else tf.convert_to_tensor(v) for k, v in kwargs.items()})
            return pred
        return predict
    else:
        return model