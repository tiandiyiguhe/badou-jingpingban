# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 200,
    "batch_size": 1024,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"F:\BaiduNetdiskDownload\八斗课程-精品班\第六周 语言模型和预训练\bert-base-chinese\bert-base-chinese",
    "vocab_size": 21128,
    "use_bert":True
}