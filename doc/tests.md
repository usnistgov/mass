## Unit testing

If you want to run the unit tests for `mass` go to your `mass` directory and do `pytest .` or equivalently, `make test`.

If you want to add tests to `mass`, you do _not_ need to use the old `unittest` framework. Simpler, more modern tests in the `pytest` style are allowed (and preferred). Put any new tests in a file somewhere inside `tests` with a name like `test_myfeature.py`, it must match the pattern `test_*.py` to be found by pytest.

On each commit to develop, the tests will be run automatically by GitHub Actions. See [results of recent tests](https://github.com/usnistgov/mass/actions).
