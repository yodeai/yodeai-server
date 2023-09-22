import unittest
from utils import exponential_backoff
from io import StringIO
import re

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
        pattern = r"exception raised; sleeping for: \d*\.?\d* seconds\nexception raised; sleeping for: \d*\.?\d* seconds"
        assert re.match(pattern, output)

if __name__ == '__main__':
    unittest.main()

