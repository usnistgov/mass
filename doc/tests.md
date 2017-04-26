## Unit testing

If you want to run the unit tests for `mass` go to your `mass` directory and do `python runtests.py` or equivalently, `make test`.

If you want to add tests to `mass`, use the `unittest` framework, and put the tests in a file somewhere inside `mass/src/mass` with a name like `test_myfeature.py`, it must match the pattern `test_*.py` to be found by `runtests.py`.

On each commit to develop, `runtests.py` will be run on `semaphoreci.com`.
