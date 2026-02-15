import os

from common.data_structures import *
from common.helpers import list_files, load_jsonl_to_dataclasses

usr_requests_dir_path = os.getenv('USR_REQUESTS_DIR_PATH', '')
black_list_for_usr_request = os.getenv('BLACK_LIST_FOR_USR_REQUESTS', '').split(',')

if __name__ == '__main__':
    usr_request_f_lst = list_files(usr_requests_dir_path)
    for usr_request_f in usr_request_f_lst:
        for prohibited_f in black_list_for_usr_request:
            if prohibited_f and prohibited_f in usr_request_f:
                print(f'==> skipping {usr_request_f}')
                continue
        result = load_jsonl_to_dataclasses(usr_request_f, Root)
