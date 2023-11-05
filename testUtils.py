import unittest
from utils import exponential_backoff
from io import StringIO
import re
import time

class TestExponentialBackoff(unittest.TestCase):
    i = 0
    def test_backoff(self):
        out = StringIO()
        @exponential_backoff(retries=3, out=out)
        def faulty_function():
            self.i = self.i+1
            if self.i == 3:
                return 1
            else:
                raise Exception("faulty!")
        faulty_function()
        output = out.getvalue().strip()
        pattern = r"Exception raised; sleeping for a backoff of: \d*\.?\d* seconds\nException raised; sleeping for a backoff of: \d*\.?\d* seconds"
        assert re.match(pattern, output)

    def test_timeout(self):
        self.i = 1
        out = StringIO()
        @exponential_backoff(retries=1, out=out)
        def faulty_function():
            if self.i == 1:
                self.i += 1
                time.sleep(6)
            else:
                time.sleep(1)
        faulty_function()
        output = out.getvalue().strip()
        pattern = r"Function timed out after 5 seconds; retrying"
        assert re.match(pattern, output)

if __name__ == '__main__':
    unittest.main()
