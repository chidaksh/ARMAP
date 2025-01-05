import time
import math
import openai
import random


class KeyPool:
    MAX_REQUESTS_PER_MINUTE = 3
    keys = []
    origin_key = ""
    index = -1

    @classmethod
    def set_all_keys(cls, keys):
        cls.keys = keys

    @classmethod
    def set_key(cls):
        random.seed(int(time.time()))
        index = random.randint(0, len(cls.keys)-1)
        wait_seconds = 0.1
        # print(f"\nWaiting {wait_seconds}s for next request.\n")
        time.sleep(wait_seconds)
        return cls.keys[index]

    @classmethod
    def reset_key(cls):
        openai.api_key = cls.origin_key

    def __enter__(self):
        return KeyPool.set_key()

    def __exit__(self, type, val, tb):
        # KeyPool.reset_key()
        pass