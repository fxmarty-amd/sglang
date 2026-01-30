import io
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_cuda_alike
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
import os

class TestOnlineQuantizationMemoryLoad(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        if os.environ.get("DISABLE_FP8", "0") == "1":
            fp8_args = []
        else:
            fp8_args = ["--quantization", "fp8"]

        other_args=fp8_args + ["--tensor-parallel-size", "1"]

        if os.environ.get("DISABLE_GRAPH", "0") == "1":
            print("Disabling CUDA Graph")
            other_args.append("--disable-cuda-graph")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=100,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # TODO: should be much higher.
        self.assertGreater(metrics["accuracy"], 0.5)