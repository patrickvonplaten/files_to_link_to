#!/usr/bin/env python3

from transformers.hf_api import HfApi

import os
import json

import sys


def main(model_identifier):
    api = HfApi()
    model_list = api.model_list()
    model_dict = [
        model_dict
        for model_dict in model_list
        if model_dict.modelId == model_identifier
    ][0]

    model_identifier = "_".join(model_identifier.split("/"))

    http = "https://s3.amazonaws.com/"
    hf_url = "models.huggingface.co/"
    config_path_aws = http + hf_url + model_dict.key
    file_name = "./{}_config.json".format(model_identifier)

    bash_command = "curl {} > {}".format(config_path_aws, file_name)
    os.system(bash_command)

    with open(file_name) as f:
        config_json = json.load(f)

    bash_command = "rm {}".format(file_name)
    os.system(bash_command)

    if 'eos_token_ids' in config_json:
        config_json.pop('eos_token_ids')

    config_json["decoder_start_token_id"] = 2

    # save config as it was saved before
    with open(file_name, "w") as f:
        json.dump(config_json, f, indent=2, sort_keys=True)

    # upload new config
    bash_command = "aws s3 cp {} s3://{}".format(file_name, hf_url + model_dict.key)
    os.system(bash_command)


if __name__ == "__main__":
    model_identifier = sys.argv[1]
    main(model_identifier)
