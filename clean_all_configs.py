#!/usr/bin/env python3
from transformers import AutoConfig
from transformers.hf_api import HfApi
import os


def clean_all_community_configs(model_list=None, do_upload=False, do_delete=True):
    api = HfApi()
    model_dict_list = api.model_list()

    if model_list is not None:
        model_dict_list = [model_dict for model_dict in model_dict_list if model_dict.modelId in model_list]
    for i, model_dict in enumerate(model_dict_list):
        model_identifier = model_dict.modelId

        hf_url = 'models.huggingface.co/'
        local_model_path = '_'.join(model_identifier.split('/'))
        path_to_config = "./{}_config.json".format(local_model_path)

        print("\n{}: Summary for {}".format(i, model_identifier))
        print(50 * '-')

        try:
            config = AutoConfig.from_pretrained(model_identifier)
        except Exception as e:  # noqa: E722
            print('CONF ERROR: {} config can not be loaded'.format(model_identifier))
            print('Message: {}'.format(e))
            print(50 * '=')
            continue

        # create temp dir
        temp_dir = path_to_config.split('.json')[0]
        if os.path.exists(temp_dir):
            os.system('rm -r {}'.format(temp_dir))
        os.mkdir(temp_dir)

        # save config locally to only use diff
        config.save_pretrained(temp_dir)
        diff_config = AutoConfig.from_pretrained(os.path.join(temp_dir, 'config.json'))

        if config != diff_config:
            print("Author: {} needs to be notified about changed conf: {}".format(model_dict.author, model_identifier))

        if do_upload is True:
            # upload new config
            bash_command = 'aws s3 cp {} s3://{}'.format(os.path.join(temp_dir, 'config.json'), hf_url + model_dict.key)
            os.system(bash_command)

            # delete saved config
        
        if do_delete is True:
            os.system('rm -r {}'.format(temp_dir))

        print(50 * '=')


#clean_all_community_configs(['patrickvonplaten/reformer-crime-and-punish'], do_upload=False, do_delete=True)
clean_all_community_configs(do_upload=True, do_delete=True)
