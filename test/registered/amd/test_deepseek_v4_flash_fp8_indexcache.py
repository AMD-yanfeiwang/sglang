"""MI35x DeepSeek-V4-Flash FP8 IndexCache test (8 GPU).

Runs the established ROCm DeepSeek-V4-Flash FP8 recipe with
``index_topk_freq=4``. Accuracy guards the producer/shared-layer data flow, and
the long-context benchmark records whether IndexCache provides a throughput
benefit on MI35x.
"""

import json
import os
import subprocess
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-amd-8-gpu-mi35x-deepseek-v4-flash",
    nightly=True,
)

DEEPSEEK_V4_FP8_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_FP8_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)
SERVER_LAUNCH_TIMEOUT = 3600
FLASHMLA_BACKEND = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")

COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_HACK_FLASHMLA_BACKEND": FLASHMLA_BACKEND,
    "SGLANG_DSV4_FP4_EXPERTS": "false",
    "AITER_BF16_FP8_MOE_BOUND": "0",
}


class TestDeepseekV4FlashFp8IndexCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--disable-radix-cache",
                "--attention-backend",
                "dsv4",
                "--max-running-requests",
                "256",
                "--page-size",
                "256",
                "--mem-fraction-static",
                "0.90",
                "--swa-full-tokens-ratio",
                "0.1",
                "--chunked-prefill-size",
                "8192",
                "--disable-shared-experts-fusion",
                "--tool-call-parser",
                "deepseekv4",
                "--reasoning-parser",
                "deepseek-v4",
                "--json-model-override-args",
                '{"index_topk_freq": 4}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                "### test_gsm8k "
                f"(deepseek-v4-flash-fp8-indexcache, {FLASHMLA_BACKEND})\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.90)

    @unittest.skipIf(
        os.environ.get("SGLANG_DSV4_ACCURACY_ONLY") == "1",
        "SGLANG_DSV4_ACCURACY_ONLY=1: accuracy-only run (skipping perf)",
    )
    def test_b_perf_64k_100(self):
        json_output = "/tmp/deepseek_v4_flash_fp8_indexcache_perf.json"
        if os.path.exists(json_output):
            os.remove(json_output)

        cmd = [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            self.base_url,
            "--batch-size",
            "1",
            "1",
            "--input-len",
            "65536",
            "--output-len",
            "100",
            "--show-report",
            f"--pydantic-result-filename={json_output}",
            "--no-append-to-github-summary",
            "--trust-remote-code",
        ]
        print(f"Running benchmark: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            self.fail(f"bench_one_batch_server failed (rc={result.returncode})")

        self.assertTrue(
            os.path.exists(json_output),
            f"Benchmark JSON output {json_output} not found",
        )
        with open(json_output) as f:
            results = json.load(f)
        self.assertTrue(results, "No benchmark results returned")
        measured = results[-1]
        summary = (
            "### test_perf_64k_100 "
            f"(deepseek-v4-flash-fp8-indexcache, {FLASHMLA_BACKEND})\n"
            f"- input throughput: {measured.get('input_throughput', 0.0):.2f} tok/s\n"
            f"- output throughput: {measured.get('output_throughput', 0.0):.2f} tok/s\n"
            f"- latency: {measured.get('latency', 0.0):.2f} s\n"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(summary)


if __name__ == "__main__":
    import sys

    sys.argv = [arg for arg in sys.argv if arg not in ("-f", "--failfast")]
    unittest.main()
