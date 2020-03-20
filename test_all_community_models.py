#!/usr/bin/env python3

from transformers import AutoConfig, AutoTokenizer
from transformers.hf_api import HfApi


def get_all_model_paths():
    api = HfApi()
    model_list = [model_dict.modelId for model_dict in api.model_list()]
    return model_list


def check_all_community_files(all_models_paths):
    for i, model_path in enumerate(all_models_paths):
        print("\n{}: Summary for {}".format(i, model_path))
        print(50 * '-')
        all_good = True

        try:
            config = AutoConfig.from_pretrained(model_path)
        except Exception as e:  # noqa: E722
            print('CONF ERROR: {} config can not be loaded'.format(model_path))
            print('Message: {}'.format(e))
            print(50 * '=')
            all_good = False
            continue

        try:
            tok = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:  # noqa: E722
            print('TOK ERROR: {} tokenizer can not be loaded'.format(model_path))
            print('Message: {}'.format(e))
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

        current_config_wrong = False

        if not pad_equal:
            all_good = False
            current_config_wrong = True
            print("PAD in Tokenizer and Config not equal for {}!".format(model_path))
            print("PAD in Tokenizer: {} | PAD in Config: {}".format(tok.pad_token_id, conf_pad_token_id))
            print(50 * "-")

        if not eos_equal:
            all_good = False
            current_config_wrong = True
            print("EOS in Tokenizer and Config not equal for {}!".format(model_path))
            print("EOS in Tokenizer: {} |EOS in Config: {}".format(tok.eos_token_id, conf_eos_token_id))
            print(50 * "-")

        if not bos_equal:
            all_good = False
            current_config_wrong = True
            print("BOS in Tokenizer and Config not equal for {}!".format(model_path))
            print("BOS in Tokenizer: {} | BOS in Config: {}".format(tok.bos_token_id, conf_bos_token_id))
            print(50 * "-")

        if all_good is True:
            print("All good!")

        if current_config_wrong is True:
            print("Config needs change!")

        print(50 * '=')


all_models_paths = get_all_model_paths()
check_all_community_files(all_models_paths)
