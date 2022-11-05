import os

import numpy as np
import tensorflow as tf

def tf2onnx(model, path, opset = 13):
    """
    - opset
    default = 13
    tflite = 16
    saved_model = 17
    """
    try:
        import tf2onnx as onnx_converter
    except Exception as e:
        print("If you want to use 'tf2onnx', please install 'tf2onnx'")
        raise e
    
    name, ext = os.path.splitext(path)
    if len(ext) < 2:
        path = "{0}{1}".format(name, ".onnx")
    
    if isinstance(model, tf.keras.Model):
        model_proto, _ = onnx_converter.convert.from_keras(model, opset = opset, output_path = path)
    elif isinstance(model, str):
        name, ext = os.path.splitext(model)
        try:
            if "tflite" in ext:
                raise
            model = tf.keras.models.load_model(model)
            return tf2onnx(model, path)
        except:
            if len(ext) < 2:
                model = "{0}{1}".format(name, ".tflite")
            #model_proto, _  = onnx_converter.convert.from_tflite(model, opset = opset, output_path = path) #model output bug
            model_proto, _ = onnx_converter.convert._convert_common(None, name = model, opset = opset, tflite_path = model, output_path = path)
    else:
        model_proto, _ = onnx_converter.convert.from_function(model, opset = opset, output_path = path)
    return path

class CalibrationDataset:#(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, data, session):
        if isinstance(session, list) or isinstance(session, tuple):
            keys = list(session)
        else:
            if isinstance(session, str):
                session = load_onnx(session, predict = False)
            keys = [inp.name for inp in session.get_inputs()]
        self.keys = keys
        del session
        data = data if isinstance(data, list) or isinstance(data, tuple) else [data]
        self.dataset = [{k:np.expand_dims(v, axis = 0) for k, v in zip(self.keys, d)} for d in zip(*data)]
        self.generator = iter(self.dataset)
    
    def get_next(self):
        return next(self.generator, None)

def onnx2quantize(path, save_path, data = None):
    try:
        import onnxruntime.quantization
    except Exception as e:
        print("If you want to use 'onnx2quantize', please install 'onnxruntime'")
        raise e
    
    name, ext = os.path.splitext(save_path)
    if len(ext) < 2:
        save_path = "{0}{1}".format(name, ".onnx")
    
    if data is None:
        onnxruntime.quantization.quantize_dynamic(path, save_path)
    else:
        calibration_dataset = CalibrationDataset(data, path) if not isinstance(data, CalibrationDataset) else data
        onnxruntime.quantization.quantize_static(path, save_path, calibration_dataset)
    return save_path

def load_onnx(path, gpu = None, n_thread = None, tensorrt = False, predict = True):
    try:
        import onnxruntime
    except Exception as e:
        print("If you want to use 'load_onnx', please install 'onnxruntime'")
        raise e
    
    name, ext = os.path.splitext(path)
    if len(ext) < 2:
        path = "{0}{1}".format(name, ".onnx")
        
    avaliable_providers = onnxruntime.get_available_providers()
    provider = ["CPUExecutionProvider"]
    if gpu is not None:
        if tensorrt and "TensorrtExecutionProvider" in avaliable_providers:
            option = {"device_id":int(gpu), "trt_fp16_enable":True}
            provider = [("TensorrtExecutionProvider", option)]
        elif "CUDAExecutionProvider" in avaliable_providers:
            option = {"device_id":int(gpu), "cudnn_conv_algo_search":"EXHAUSTIVE"}
            #option["gpu_mem_limit"] = int(gpu_mem_limit)
            provider = [("CUDAExecutionProvider", option)]

    option = None
    if n_thread is not None:
        option = onnxruntime.SessionOptions()
        option.inter_op_num_threads = int(n_thread)
        option.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    session = onnxruntime.InferenceSession(path, sess_options = option, providers = provider)
    if predict:
        input_keys = [node.name for node in session.get_inputs()]
        def predict(*args, **kwargs):
            args = {k:v for k, v in zip(input_keys[:len(args)], args)}
            kwargs.update(args)
            pred = session.run(None, kwargs)
            if len(pred) == 0:
                pred = None
            elif len(pred) == 1:
                pred = pred[0]
            return pred
        return predict
    else:
        return session