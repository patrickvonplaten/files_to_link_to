#!/usr/bin/env python3

from transformers.hf_api import HfApi
from transformers import AutoConfig, AutoTokenizer

import os
import json


def download(config_path_aws, model_identifier):

    file_name = "./{}_config.json".format(model_identifier)
    bash_command = "curl {} > {}".format(config_path_aws, file_name)
    os.system(bash_command)

    with open(file_name) as f:
        config_json = json.load(f)

    bash_command = 'rm {}'.format(file_name)
    os.system(bash_command)

    return file_name, config_json


api = HfApi()
model_list = api.model_list()

for model_dict in model_list:
    model_identifier = model_dict.modelId

    try:
        config = AutoConfig.from_pretrained(model_identifier)
        tok = AutoTokenizer.from_pretrained(model_identifier)
    except Exception:
        continue

    DefaultClass = config.__class__
    default_config = DefaultClass()

    http = 'https://s3.amazonaws.com/'
    hf_url = 'models.huggingface.co/'
    config_path_aws = http + hf_url + model_dict.key

    model_identifier = '_'.join(model_identifier.split('/'))

    path_to_config, config_json = download(config_path_aws, model_identifier)
    is_change = False

    if 'eos_token_ids' in config_json:
        del config_json['eos_token_ids']
        is_change = True

    if 'pad_token_id' in config_json and config_json['pad_token_id'] != tok.pad_token_id:
        del config_json['pad_token_id']
        is_change = True

    if 'bos_token_id' in config_json and config_json['bos_token_id'] != tok.bos_token_id:
        del config_json['bos_token_id']
        is_change = True

    if 'eos_token_id' in config_json and config_json['eos_token_id'] != tok.eos_token_id:
        del config_json['eos_token_id']
        is_change = True

    if 'pad_token_id' not in config_json and default_config.pad_token_id != tok.pad_token_id:
        config_json['pad_token_id'] = tok.pad_token_id
        is_change = True

    if 'bos_token_id' not in config_json and default_config.bos_token_id != tok.bos_token_id:
        config_json['bos_token_id'] = tok.bos_token_id
        is_change = True

    if 'eos_token_id' not in config_json and default_config.eos_token_id != tok.eos_token_id:
        config_json['eos_token_id'] = tok.eos_token_id
        is_change = True

    if is_change:

        if model_dict.author is not None and isinstance(model_dict.author, str):
            print(model_dict.author + ' - ' + model_dict.modelId)
        else:
            print("No Author - " + model_dict.modelId)
        # save config as it was saved before
        with open(path_to_config, 'w') as f:
            json.dump(config_json, f, indent=2, sort_keys=True)

        # upload new config
        bash_command = 'aws s3 cp {} s3://{}'.format(path_to_config, hf_url + model_dict.key)
        os.system(bash_command)

        # delete saved config
        os.system('rm {}'.format(path_to_config))
