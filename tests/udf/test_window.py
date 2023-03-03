import os
import orjson
import unittest
from unittest.mock import patch, Mock

from pynumaflow.function._dtypes import DROP

from numaprom import tools
from numaprom._constants import TESTS_DIR
from numaprom.entities import StreamPayload
from tests.tools import get_datum, get_stream_data, mockenv, mock_configs
from tests import redis_client, window

DATA_DIR = os.path.join(TESTS_DIR, "resources", "data")
STREAM_DATA_PATH = os.path.join(DATA_DIR, "stream.json")


@patch.object(tools, "get_all_configs", Mock(return_value=mock_configs()))
class TestWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_stream = get_stream_data(STREAM_DATA_PATH)

    def tearDown(self) -> None:
        redis_client.flushall()

    def test_window(self):
        for idx, data in enumerate(self.input_stream):
            _out = window("", get_datum(data))
            if not _out.items()[0].key == DROP:
                _out = _out.items()[0].value.decode("utf-8")
                payload = StreamPayload(**orjson.loads(_out))
                keys = list(payload.composite_keys.values())
                if "rollout" in keys[1]:
                    self.assertEqual(len(keys), 3)
                else:
                    self.assertEqual(len(keys), 2)

    @mockenv(BUFF_SIZE="1")
    def test_window_err(self):
        with self.assertRaises(ValueError):
            for data in self.input_stream:
                window("", get_datum(data))

    def test_window_drop(self):
        for _d in self.input_stream:
            out = window("", get_datum(_d))
            self.assertEqual(DROP, out.items()[0].key)
            break


if __name__ == "__main__":
    unittest.main()
