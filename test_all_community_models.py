#!/usr/bin/env python3

from transformers import AutoConfig
from transformers import AutoTokenizer

import sys


def read_in_file(path_to_file):
    with open(path_to_file, 'r') as file_to_read:
        lines_in_file = file_to_read.readlines()
        lines_in_file = [line.strip() for line in lines_in_file]
        return lines_in_file


def check_all_community_files(all_models_paths):
    for i, model_path in enumerate(all_models_paths):
        print("\n{}: Summary for {}".format(i, model_path))
        print(50 * '-')
        all_good = True

        try:
            config = AutoConfig.from_pretrained(model_path)
            tok = AutoTokenizer.from_pretrained(model_path)
        except:  # noqa: E722
            print('ERROR: {} config or tokenizer can not be loaded'.format(model_path))
            print(50 * '=')
            all_good = False
            continue

        DefaultClass = config.__class__
        default_config = DefaultClass()

        if hasattr(config, 'eos_token_ids'):
            all_good = False
            print('TODO_1: In {} the eos_token_ids has to be removed'.format(model_path))
            eos_token_ids = config.eos_token_ids[0] if type(config.eos_token_ids) is list else config.eos_token_ids
            if hasattr(default_config, 'eos_token_id') and eos_token_ids != default_config.eos_token_id:
                print('TODO_2: In {} eos_token_ids is {} but default eos_token_id is {} - Adapt on AWS'.format(model_path, eos_token_ids, default_config.eos_token_id))
            conf_eos_token_id = eos_token_ids
        else:
            conf_eos_token_id = default_config.eos_token_id

        if hasattr(config, 'bos_token_id'):
            conf_bos_token_id = config.bos_token_id
        else:
            conf_bos_token_id = default_config.bos_token_id

        if hasattr(config, 'pad_token_id'):
            conf_pad_token_id = config.pad_token_id
        else:
            conf_pad_token_id = default_config.pad_token_id

        pad_equal = tok.pad_token_id == conf_pad_token_id
        eos_equal = tok.eos_token_id == conf_eos_token_id
        bos_equal = tok.bos_token_id == conf_bos_token_id

        print(50 * '-')

        if not pad_equal:
            all_good = False
            print("PAD in Tokenizer and Config not equal for {}!".format(model_path))
            print("PAD in Tokenizer: {} | PAD in Config: {}".format(tok.pad_token_id, conf_pad_token_id))
            print(50 * "-")

        if not eos_equal:
            all_good = False
            print("EOS in Tokenizer and Config not equal for {}!".format(model_path))
            print("EOS in Tokenizer: {} |EOS in Config: {}".format(tok.eos_token_id, conf_eos_token_id))
            print(50 * "-")

        if not bos_equal:
            all_good = False
            print("BOS in Tokenizer and Config not equal for {}!".format(model_path))
            print("BOS in Tokenizer: {} | BOS in Config: {}".format(tok.bos_token_id, conf_bos_token_id))
            print(50 * "-")

        if all_good is True:
            print("All good!")

        print(50 * '=')


all_models_paths = read_in_file(sys.argv[1])
check_all_community_files(all_models_paths)
