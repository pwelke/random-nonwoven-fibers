import os
import requests
import hashlib
import gzip

url = 'https://mlai.cs.uni-bonn.de/publications/kalofolias2021-sdm.pdf'
expected_checksum = '196c7665dfe6d86df3b3df72a767267e'
zipfile = os.path.join('.', 'data', 'zipped_data.tar.gz')
chunk_size = 1024

def _download_data():
    r = requests.get(url, stream = True)
    h = hashlib.md5()

    with open(zipfile, 'wb') as f:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                h.update(chunk)

    checksum = h.hexdigest()

    if checksum != expected_checksum:
        raise ValueError(f'Error: Checksum mismatch: Expected checksum is {expected_checksum}, but we got {checksum}')

def correct_file_exists():
    '''returns true if the file is present and has the correct checksum
    returns false if checksum is wrong or reading file produces IOError'''
    h = hashlib.md5()
    try:
        with open(zipfile, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if len(chunk) > 0:
                    h.update(chunk)
                else:
                    break
    except IOError:
        # if file does not exist or similar, return false
        return False

    checksum = h.hexdigest()
    return checksum == expected_checksum
        

def download_and_unpack():
    if not correct_file_exists():
        print(f'Downloading data file from {url}')
        _download_data()
    else:
        print('everything is fine')
    

download_and_unpack()
    