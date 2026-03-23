# Find and fix tritonbench testx errors

## 1. Find the testx errors

Use the testx command to find the testx errors related to tritonbench:

```bash
testx tests search -n "tritonbench/test/test_gpu:test_gpu" -s problematic 
```

Output how many failed tests there are, names of the failed tests, and their test IDs.

Example output:

```
<NUMBERS> failed (problematic) tests found. All have status PROBLEMATIC and trunk state DISABLED_FAILING.

From test_gpu target (8 tests):
- <NAME> (test_id: <TEST_ID>)
- ...

From test_gpu_triton_beta target (6 tests):
- <NAME>: (test_id: <TEST_ID>)
- ...
```

## 2. Find the run id of the most recent failed test ID and download the test fail details

For each of the failed test above, use the command below to find the most recent failed test run id:

```bash
testx tests overview <TEST_ID>
```

For each of the most recent failed test run id, use the command below to find its test fail details:

```bash
testx --as-json results search <RUN_ID> --statuses FAILED --limit 1
```

Only keep the failed tests that have the most recent failed test in the past 3 days. Keep the time out tests.
Replace the <TEST_ID> with link https://www.internalfb.com/intern/test/<TEST_ID>.
For the most recent failed run ID, do not keep its "SC:" prefix, if it exists.
Save the test fail details in a file.

Example output:

```
<NUMBERS> failed (problematic) tests found. All have status PROBLEMATIC and trunk state DISABLED_FAILING.

From test_gpu target (8 tests):
- <NAME> (test_id: <TEST_ID>, most recent failed test run id: <RUN_ID>, test fail details: <FAIL_DETAILS_FILE_PATH>)
- ...

From test_gpu_triton_beta target (6 tests):
- <NAME>: (test_id: <TEST_ID>, most recent failed test run id: <RUN_ID>, test fail details: <FAIL_DETAILS_FILE_PATH>)
- ...

Timeout tests:
- <NAME> (test_id: <TEST_ID>, most recent failed test run id: <RUN_ID>)
- ...

Other tests:
- <NAME> (test_id: <TEST_ID>, most recent failed test run id: <RUN_ID>)
- ...
```

It is possible that there is no "Other tests" section, if all the tests are either timeout or can find the most recent failed test run id.

## 3. Group tests by their test fail details

Read all download test fail details files and group them by their error messages.

Example output:

```
<NUMBERS> failed (problematic) tests found. All have status PROBLEMATIC and trunk state DISABLED_FAILING.

Error message 1:
<Error message>

- <TEST NAME 1>
- ...


Error message 2:
<Error message>

- <TEST NAME 2>
- ...

...

Time out Tests:
- <TEST NAME 3>
- ...

Other tests (unknown error message or could not find most recent failed test run id):
- <TEST NAME 5>
- ..
```

## 4. Attempt to fix the tests

For each test that has a known error message or time out, try to fix it.
Create a Diff stack for the fixes.

Prioritize fixing time out tests first, then fixing tests with known error messages.
To fix a timeout test, you can add this test to `tritonbench/test/test_gpu/fb/skip_tests.py`.

For example, if `test_gpu_tritonbench_reduction_gemm` is a timeout test, you can add the following to `tritonbench/test/test_gpu/fb/skip_tests.py`

```yaml
# skip reduction_gemm: test timeout
reduction_gemm:
```
