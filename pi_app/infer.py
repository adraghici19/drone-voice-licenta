import os
import numpy as np
import onnxruntime as ort


class LiteVoiceNetONNX:
    def __init__(self, cfg):
        self.cfg = cfg
        model_path = self._resolve_model_path(cfg)
        print("[infer] loading {}".format(model_path))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        self._h0 = np.zeros(
            (cfg.gru_layers, 1, cfg.gru_hidden), dtype=np.float32
        )

    @staticmethod
    def _resolve_model_path(cfg) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        fp32 = os.path.normpath(os.path.join(here, cfg.onnx_fp32))
        int8 = os.path.normpath(os.path.join(here, cfg.onnx_int8))
        if cfg.use_int8 and os.path.exists(int8):
            return int8
        if os.path.exists(fp32):
            return fp32
        raise FileNotFoundError(
            "No ONNX model found.\n  Looked for:\n    {}\n    {}".format(int8, fp32)
        )

    def run(self, features: np.ndarray):
        x = features[np.newaxis, ...].astype(np.float32)
        mask, vad, kws, _ = self.sess.run(
            None, {"features": x, "h_in": self._h0}
        )
        mask = mask[0, 0]
        vad = vad[0, :, 0]
        kws = np.exp(kws[0])
        return mask, vad, kws
