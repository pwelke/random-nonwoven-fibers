import os
import requests
import hashlib
import gzip

resources = {'labeled': {
                'url': '', 
                'checksum': 'd1fd520c883f971276df847d6eae2533', 
                'filename': os.path.join('.', 'data', 'labeled.tar.gz')},
             'unlabeled': {
                 'url': '', 
                 'checksum': '813248147edb542e6d089a67aa5261ea', 
                 'filename': os.path.join('.', 'data','unlabeled.tar.gz')},
             'dummy': {
                 'url': 'https://mlai.cs.uni-bonn.de/publications/kalofolias2021-sdm.pdf',
                 'checksum': '196c7665dfe6d86df3b3df72a767267e',
                 'filename': os.path.join('.', 'data','dummy.pdf')}
             }

chunk_size = 1024

def _download_data(dataset):
    r = requests.get(resources[dataset]['url'], stream = True)
    h = hashlib.md5()

    with open(resources[dataset]['filename'], 'wb') as f:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                h.update(chunk)

    checksum = h.hexdigest()

    if checksum != resources[dataset]['checksum']:
        raise ValueError(f'Error: Checksum mismatch: Expected checksum is {resources[dataset]["checksum"]}, but we got {checksum}')

def correct_file_exists(dataset):
    '''returns true if the file is present and has the correct checksum
    returns false if checksum is wrong or reading file produces IOError'''
    h = hashlib.md5()
    try:
        with open(resources[dataset]['filename'], 'rb') as f:
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
    return checksum == resources[dataset]['checksum']
        

def download_and_unpack(dataset):
    if not correct_file_exists(dataset):
        print(f'Downloading data file for dataset {dataset} from {resources[dataset]["url"]}')
        _download_data()
    else:
        print('everything is fine')
    

download_and_unpack('labeled')
download_and_unpack('unlabeled')
    