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


def clean_token_ids(model_dict, config, model_identifier, config_json):

    try:
        config = AutoConfig.from_pretrained(model_identifier)
        tok = AutoTokenizer.from_pretrained(model_identifier)
    except Exception:
        return False

    DefaultClass = config.__class__
    default_config = DefaultClass()

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

    return is_change


def change_model_list(change_fn, model_list=None, do_upload=False, key_word=None):
    api = HfApi()
    model_dict_list = api.model_list()

    if model_list is not None:
        model_dict_list = [model_dict for model_dict in model_dict_list if model_dict.modelId in model_list]

    if key_word is not None:
        model_dict_list = [model_dict for model_dict in model_dict_list if key_word in model_dict.modelId]

    for model_dict in model_dict_list:
        model_identifier = model_dict.modelId

        print("model_identifier")

        http = 'https://s3.amazonaws.com/'
        hf_url = 'models.huggingface.co/'
        config_path_aws = http + hf_url + model_dict.key
        model_identifier = '_'.join(model_identifier.split('/'))
        path_to_config, config_json = download(config_path_aws, model_identifier)

        config_json = change_fn(config_json)

        # save config as it was saved before
        with open(path_to_config, 'w') as f:
            json.dump(config_json, f, indent=2, sort_keys=True)

        if do_upload is True:
            # upload new config
            bash_command = 'aws s3 cp {} s3://{}'.format(path_to_config, hf_url + model_dict.key)
            os.system(bash_command)

            # delete saved config
            os.system('rm {}'.format(path_to_config))


def bart_prefix(config_json):
    del config_json['intermediate_size']
    del config_json['chunk_length']
    config_json['lsh_attn_chunk_length'] = 64
    config_json['local_attn_chunk_length'] = 64
    return config_json


def set_hash_seed(config_json):
    del config_json['sinusoidal_pos_embds']
    del config_json['type_vocab_size']
    config_json['model_type'] = "reformer"
    return config_json


def change_model(config_json):
    config_json["encoder"]["model_type"] = 'bert-generation'
    config_json["decoder"]["model_type"] = 'bert-generation'
    return config_json


def change_single_model(config_json):
    config_json["model_type"] = 'bert-generation'
    return config_json


def print_padding_id(config_json):
    if 'pad_token_id' in config_json:
        print("pad_token_id:", config_json['pad_token_id'])


change_model_list(change_model, do_upload=True, key_word="google/bert2bert")
change_model_list(change_model, do_upload=True, key_word="google/roberta2roberta")
change_model_list(change_single_model, do_upload=True, key_word="google/bert_for_seq_generation")
