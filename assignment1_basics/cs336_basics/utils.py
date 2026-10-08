PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class AutoKeyDictWrapper:
    def __init__(self):
        self.dict: dict[int, bytes] = {}
        self._key: int = 0

    def add_one(self, value: bytes):
        self.dict[self._key] = value
        self._key += 1

    def add_list(self, values: list[bytes]):
        increment = len(values)
        
        for i, b in enumerate(values, start=self._key):
            self.dict[i] = b
        
        self._key += increment


def bytes_to_unicode() -> dict[int: str]:
    """
    GPT-2 style remap function to transfer illegal utf-8 bytes to 
    printable unicode characters.
    """
    valid_bytes = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    unicode_map = valid_bytes[:]

    n = 0
    for i in range(256):
        if i not in valid_bytes:
            valid_bytes.append(i)
            unicode_map.append(256 + n)
            n +=1

    unicode_map = [chr(c) for c in unicode_map]
    return dict(zip(valid_bytes, unicode_map))

        
        