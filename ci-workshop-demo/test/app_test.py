import requests
import unittest

class TestAppEndpoint(unittest.TestCase):
    
    def test_endpoint(self):
        query_params = '?date=2017-06-14&item_nbr=99197';
        resp = requests.get('http://localhost:5005/prediction' + query_params);

        assert resp.status_code == 200

if __name__ == "__main__":
    unittest.main()