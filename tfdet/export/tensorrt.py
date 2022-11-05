import os
import tempfile

import numpy as np
import tensorflow as tf

def tf2trt(model, save_path, dtype = "FP32", memory_limit = 1, dynamic_batch = False, n_thread = 1, data = None):
    """
    dtype = dtype in [tf.int8, tf.float16, tf.float32, np.int8, np.float16, np.float32, 'INT8', 'FP16', 'FP32']
    dynamic_batch = static batch size or dynamic batch size
    memory_limit = x.xx gigabyte
    data = [np.array, ...]
    """
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
    except Exception as e:
        print("If you want to use 'tf2trt', please run it on nvidia gpu")
        raise e
    
    if dtype not in [tf.int8, tf.float16, tf.float32, np.int8, np.float16, np.float32, "INT8", "FP16", "FP32"]:
        raise ValueError("unknown dtype '{0}'".format(dtype))
        
    dtype_converter = {tf.int8:"INT8", tf.float16:"FP16", tf.float32:"FP32", np.int8:"INT8", np.float16:"FP16", np.float32:"FP32"}
    dtype = dtype_converter[dtype] if dtype in dtype_converter else dtype
    
    with tempfile.TemporaryDirectory() as temp_path:
        if isinstance(model, tf.keras.Model):
            model.save(temp_path)
            model = temp_path
        
        trt_param = trt.TrtConversionParams(precision_mode = dtype if data is None else "INT8", 
                                            max_workspace_size_bytes = int(round((1 << 30) * memory_limit)),
                                            maximum_cached_engines = n_thread)
        converter = trt.TrtGraphConverterV2(input_saved_model_dir = model,
                                            conversion_params = trt_param,
                                            use_dynamic_shape = dynamic_batch, 
                                            dynamic_shape_profile_strategy = "Optimal")
        if data is not None:
            data = data if isinstance(data, (list, tuple)) else [data]
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

def onnx2trt(path, save_path, dtype = "FP32", memory_limit = 1, data = None, verbose = True):
    try:
        import tensorrt as trt
    except Exception as e:
        print("If you want to use 'onnx2trt', please install 'tensorrt'")
        raise e
        
    if dtype not in [tf.int8, tf.float16, tf.float32, np.int8, np.float16, np.float32, "INT8", "FP16", "FP32"]:
        raise ValueError("unknown dtype '{0}'".format(dtype))
        
    dtype_converter = {tf.int8:"INT8", tf.float16:"FP16", tf.float32:"FP32", np.int8:"INT8", np.float16:"FP16", np.float32:"FP32"}
    dtype = dtype_converter[dtype] if dtype in dtype_converter else dtype
        
    name, ext = os.path.splitext(path)
    if len(ext) < 2:
        path = "{0}{1}".format(name, ".onnx")
    name, ext = os.path.splitext(save_path)
    if len(ext) < 2:
        save_path = "{0}{1}".format(name, ".trt")
    
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    trt.init_libnvinfer_plugins(logger, "")
    
    network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(network_flag) as network, trt.OnnxParser(network, logger) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = int(round((1 << 30) * memory_limit))
        
        with open(path, "rb") as file:
            if not parser.parse(file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        builder.max_batch_size = inputs[-1].shape[0] #1
        
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if data is not None:# and builder.platform_has_fast_int8:
            if True:#builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            data = data if isinstance(data, (list, tuple)) else [data]
            def representative_dataset():
                for ds in zip(*data):
                    yield [np.expand_dims(d, axis = 0) for d in ds]
            config.int8_calibrator.set_image_batcher(representative_dataset)
        else:
            if dtype == "FP16": #and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif dtype == "INT8": #and builder.platform_has_fast_int8:
                if True:#builder.platform_has_fast_fp16: #https://github.com/Linaom1214/TensorRT-For-YOLO-Series
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.INT8)
        
        engine_bytes = None
        try:
            engine_bytes = builder.build_serialized_network(network, config)
        except AttributeError:
            engine = builder.build_engine(network, config)
            engine_bytes = engine.serialize()
            del engine
        if engine_bytes is not None:
            with open(save_path, "wb") as file:
                file.write(engine_bytes)
            return save_path
    
def add_trt_nms(path, save_path, dtype = None,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05,
                ignore_label = 0,
                raw_output = True,
                nms = True,
                name = "trt_nms"):
    """
    To convert onnx model to tensorrt, add tensorrt nms.
    path = onnx model path
    """
    try:
        import onnx
        import onnx_graphsurgeon as gs
    except Exception as e:
        print("If you want to use 'add_trt_nms', please install 'onnx' and 'onnx_graphsurgeon'")
        raise e
    
    if dtype not in [None, tf.float16, tf.float32, np.float16, np.float32, "FP16", "FP32"]:
        raise ValueError("unknown dtype '{0}'".format(dtype))
        
    filename, ext = os.path.splitext(path)
    if len(ext) < 2:
        path = "{0}{1}".format(filename, ".onnx")
    filename, ext = os.path.splitext(save_path)
    if len(ext) < 2:
        save_path = "{0}{1}".format(filename, ".onnx")
        
    graph = gs.import_onnx(onnx.load(path))
    graph.fold_constants()
    graph.cleanup().toposort()

    #https://github.com/WongKinYiu/yolov7
    for _ in range(3):
        count_before = len(graph.nodes)
        graph.cleanup().toposort()
        for node in graph.nodes:
            for o in node.outputs:
                o.shape = None
        model = gs.export_onnx(graph)
        model = onnx.shape_inference.infer_shapes(model)
        graph = gs.import_onnx(model)
        graph.fold_constants(fold_shapes=True)
        count_after = len(graph.nodes)
        if count_before == count_after:
            break
    
    batch_size = graph.inputs[0].shape[0]
    if np.any([s < 0 for s in graph.inputs[0].shape[:3]]):
        raise ValueError("input layer should be written like 'tf.keras.layers.Input(shape = (1024, 1024, 3), batch_size = 1)'.")
        
    dtype_converter = {tf.float16:np.float16, tf.float32:np.float32, "FP16":np.float16, "FP32":np.float32}
    dtype = dtype_converter[dtype] if dtype in dtype_converter else dtype
    dtype = graph.outputs[0].dtype.type if dtype is None else dtype

    num_dets = gs.Variable(name = "num_dets", dtype = np.int32, shape = [batch_size, 1])
    det_scores = gs.Variable(name = "det_scores", dtype = dtype, shape = [batch_size, proposal_count])
    det_bboxes = gs.Variable(name = "det_bboxes", dtype = dtype, shape = [batch_size, proposal_count, 4])

    logits, proposal = graph.outputs[:2]
    remain = graph.outputs[2:]
    if nms:
        if True:
            attrs = {
                "plugin_version": "1",
                "background_class":ignore_label,
                "max_output_boxes":proposal_count,
                "score_threshold":score_threshold,
                "iou_threshold":iou_threshold,
                "score_activation":False,
                "box_coding":0,
            }

            det_classes = gs.Variable(name = "det_classes", dtype = np.int32, shape = [batch_size, proposal_count])
            nms_node = gs.Node(op = "EfficientNMS_TRT", name = name, attrs = attrs, inputs = [proposal, logits] + remain, outputs = [num_dets, det_bboxes, det_scores, det_classes])
        else:
            if len(proposal.shape) < 4:
                shape = [*proposal.shape[:2], 1, proposal.shape[-1]]
                proposal_unqueeze = gs.Variable("unsqueeze_{0}".format(proposal.name), shape = shape, dtype = dtype)
                axes = gs.Constant(name = "unsqueeze_axes_{0}".format(proposal.name), values = np.array(-2, dtype = np.int64))
                unsqueeze_node = gs.Node(op = "Unsqueeze", name = proposal_unqueeze.name, inputs = [proposal, axes], outputs = [proposal_unqueeze])
                graph.nodes.append(unsqueeze_node)
                #graph.outputs = [logits, proposal_unqueeze] + remain
                proposal = proposal_unqueeze

            attrs = {
                "shareLocation": True if len(proposal.shape) < 4 or proposal.shape[-2] != logits.shape[-1] else False,
                "background_class":ignore_label,
                "numClasses":logits.shape[-1],
                "topK":4096, #4096 is max val.
                "keepTopK": proposal_count, 
                "scoreThreshold":score_threshold,
                "iouThreshold":iou_threshold,
                "isNormalized":True,
                "clipBoxes":False,
                #"scoreBits":16,
            }

            det_classes = gs.Variable(name = "det_classes", dtype = dtype, shape = [batch_size, proposal_count])
            nms_node = gs.Node(op = "BatchedNMS_TRT", name = name, attrs = attrs, inputs = [proposal, logits] + remain, outputs = [num_dets, det_bboxes, det_scores, det_classes])
        graph.nodes.append(nms_node)
    else:
        proposal_count = proposal.shape[1]
        num_dets = gs.Variable(name = "num_dets", dtype = np.int32, shape = [batch_size, 1])
        det_scores = gs.Variable(name = "det_scores", dtype = dtype, shape = [batch_size, proposal_count])
        #det_bboxes = gs.Variable(name = "det_bboxes", dtype = dtype, shape = [batch_size, proposal_count, 4])
        det_classes = gs.Variable(name = "det_classes", dtype = np.int64, shape = [batch_size, proposal_count])
        
        argmax_node = gs.Node(op = "ArgMax", name = "{0}/{1}".format(name, "det_classes"), attrs = {"axis":-1, "keepdims":False}, inputs = [logits], outputs = [det_classes])
        graph.nodes.append(argmax_node)
        max_node = gs.Node(op = "ReduceMax", name = "{0}/{1}".format(name, "det_scores"), attrs = {"axes":[-1], "keepdims":False}, inputs = [logits], outputs = [det_scores])
        graph.nodes.append(max_node)
        nonzero_node = gs.Node(op = "NonZero", name = "{0}/{1}".format(name, "num_dets"), attrs = {}, inputs = [det_classes], outputs = [num_dets])
        graph.nodes.append(nonzero_node)
        det_bboxes = proposal
    
    if raw_output:
        graph.outputs = [num_dets, det_bboxes, det_scores, det_classes] #It is impossible to change the order.
    else:
        name = "{0}/postprocess/{1}".format(name, "{0}")
        
        det_scores_unsqueeze = gs.Variable(name = "det_scores_unsqueeze", dtype = dtype, shape = [*det_scores.shape[:2], 1])
        axes = gs.Constant(name = "det_scores_unsqueeze_axes", values = np.array(-1, dtype = np.int64))
        unsqueeze_node = gs.Node(op = "Unsqueeze", name = name.format("det_scores_unsqueeze"), attrs = {}, inputs = [det_scores, axes], outputs = [det_scores_unsqueeze])
        graph.nodes.append(unsqueeze_node)
        
        det_classes_unsqueeze = gs.Variable(name = "det_classes_unsqueeze", dtype = det_classes.dtype, shape = [*det_classes.shape[:2], 1])
        axes = gs.Constant(name = "det_classes_unsqueeze_axes", values = np.array(-1, dtype = np.int64))
        unsqueeze_node = gs.Node(op = "Unsqueeze", name = name.format("det_classes_unsqueeze"), attrs = {}, inputs = [det_classes, axes], outputs = [det_classes_unsqueeze])
        graph.nodes.append(unsqueeze_node)
        
        det_onehots = gs.Variable(name = "det_onehots", dtype = np.bool, shape = [*det_classes.shape[:2], logits.shape[-1]])
        value = gs.Constant(name = "det_onehots_value", values = np.reshape(np.arange(logits.shape[-1]).astype(det_classes.dtype), (1, 1, -1)))
        equal_node = gs.Node(op = "Equal", name = name.format("det_onehots"), attrs = {}, inputs = [value, det_classes_unsqueeze], outputs = [det_onehots])
        graph.nodes.append(equal_node)
        
        det_onehots_cast = gs.Variable(name = "det_onehots_cast", dtype = dtype, shape = det_onehots.shape)
        cast_node = gs.Node(op = "Cast", name = name.format("det_onehots_cast"), attrs = {"to":dtype}, inputs = [det_onehots], outputs = [det_onehots_cast])
        graph.nodes.append(cast_node)
        
        det_logits = gs.Variable(name = "det_logits", dtype = dtype, shape = det_onehots_cast.shape)
        logits_node = gs.Node(op = "Mul", name = name.format("det_logits"), attrs = {}, inputs = [det_onehots_cast, det_scores_unsqueeze], outputs = [det_logits])
        graph.nodes.append(logits_node)
        
        graph.outputs = [det_bboxes, det_logits] #It is impossible to change the order.
        
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, save_path)
    return save_path

def load_trt(path, predict = True):
    """
    - path
    file_path or "trt" in ext > load_trt_onnx
    dir_path > load_trt_tf
    """
    if not os.path.isdir(path) or "trt" in os.path.splitext(path)[1] or os.path.exists(path + ".trt"):
        return load_trt_onnx(path, predict)
    else:
        return load_trt_tf(path, predict)

def load_trt_tf(path, predict = True):
    """
    path = tf2trt model_dir_path
    """
    try:
        from tensorflow.python.saved_model import signature_constants, tag_constants
        from tensorflow.python.framework import convert_to_constants
    except Exception as e:
        print("If you want to use 'load_trt_tf', please run it on nvidia gpu")
        raise e
    
    model = tf.saved_model.load(path, tags = [tag_constants.SERVING])
    if predict:
        signature_runner = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_keys = [tensor.name.split(":")[0] for tensor in signature_runner.inputs]
        signature_runner = convert_to_constants.convert_variables_to_constants_v2(signature_runner)
        def predict(*args, **kwargs):
            args = {k:v for k, v in zip(input_keys[:len(args)], args)}
            kwargs.update(args)
            pred = signature_runner(**{k:v if tf.is_tensor(v) else tf.convert_to_tensor(v) for k, v in kwargs.items()})
            return pred
        return predict
    else:
        return model
    
def load_trt_onnx(path, predict = True):
    """
    path = onnx2trt model_path
    """
    try:
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda
    except Exception as e:
        print("If you want to use 'load_trt_onnx', please install 'tensorrt' and 'pycuda'")
        raise e
        
    name, ext = os.path.splitext(path)
    if len(ext) < 2:
        path = "{0}{1}".format(name, ".trt")
    
    with open(path, "rb") as file, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        trt.init_libnvinfer_plugins(None, "")
        engine = runtime.deserialize_cuda_engine(file.read())
    context = engine.create_execution_context()
    
    model_info = {"inputs":[],
                  "outputs":[],
                  "allocations":[]}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
        shape = context.get_binding_shape(i)

        if engine.binding_is_input(i) and shape[0] < 0: #input_node and dynamic shape
            profile_shape = engine.get_profile_shape(0, name)
            context.set_binding_shape(i, profile_shape[-1])
            shape = context.get_binding_shape(i)

        size = dtype.itemsize
        for s in shape:
            size *= s
        allocation = cuda.mem_alloc(size)

        binding = {"index":i,
                   "name":name,
                   "dtype":dtype,
                   "shape":list(shape),
                   "allocation":allocation,
                   "host_allocation":None if engine.binding_is_input(i) else np.zeros(shape, dtype)}

        model_info["allocations"].append(allocation)
        if engine.binding_is_input(i):
            model_info["inputs"].append(binding)
        else:
            model_info["outputs"].append(binding)
    if predict:
        input_keys = [node["name"] for node in model_info["inputs"]]
        def predict(*args, **kwargs):
            args = {k:v for k, v in zip(input_keys[:len(args)], args)}
            kwargs.update(args)
            for node in model_info["inputs"]:
                cuda.memcpy_htod(node["allocation"], kwargs[node["name"]]) #from host to gpu
            context.execute_v2(model_info["allocations"]) #inference
            for node in model_info["outputs"]:
                cuda.memcpy_dtoh(node["host_allocation"], node["allocation"]) # from gpu to host
            pred = [node["host_allocation"] for node in model_info["outputs"]]
            if len(pred) == 0:
                pred = None
            elif len(pred) == 1:
                pred = pred[0]
            return pred
        return predict
    else:
        return model_info