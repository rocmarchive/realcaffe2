## Running operator tests in caffe2

After building the source following the build instructions, caffe2 operators can be tested using python test scripts. In your build directory, please navigate to caffe2/python/opertor_test to see a list of available operator tests.

## Run all the tests at once
All the tests can be run at once using pytest module. If you do not have a pytest module, install it by

```
pip install pytest

```

To run the tests please navigate to build/bin directory.

```
cd <caffe2_home>/build/bin
```
From inside the bin directory operator tests can be invoked using pytest:

```
pytest ../caffe2/python/operator_test/ 

```
This command runs all the operator tests in operator_test directory. To ignore runnning any tests, --ignore flag can be used.

```
pytest ../caffe2/python/operator_test/ --ignore <name_of_test_to_ignore>

```
Multiple --ignore arguments can be passed to ingore mulitple tests. Please read pytest documentation to explore more options at https://docs.pytest.org/en/latest/usage.html 

 
## Run individual tests

To run each test separately, navigate to build\bin directory and run python test

```
cd <caffe2_home>/build/bin
python ../caffe2/operator_test/<test_name>

```

When running tests separately, please ensure that the test script has a main function, else add the following lines at the end of the script.
```
if __name__ == "__main__":
    import unittest
    unittest.main()

```


