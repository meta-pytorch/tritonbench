import json

from expecttest import assert_expected_inline
from generate_bisect_parameters import generate_benchmark_matrix


def test_generate_benchmark_matrix():
    # All combinations, no duplication
    triton_channel = "triton-main"
    runner = "h100"
    output = json.dumps(generate_benchmark_matrix(triton_channel, runner), indent=2)
    assert_expected_inline(
        output,
        """\
{
  "include": [
    {
      "runner": "gcp-h100-runner",
      "triton_channel": "triton-main"
    }
  ]
}""",
    )
