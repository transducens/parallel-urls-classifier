
import os
import logging

import parallel_urls_classifier.utils.utils as utils

import torch.nn as nn
import transformers

logger = logging.getLogger("parallel_urls_classifier")

# Adapted from https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=aVX5hFlzmLka
class MultitaskModel(transformers.PreTrainedModel):
    config_class = None
    #is_parallelizable = True # TODO ?
    base_model_prefix = "encoder"
    main_input_name = "input_ids"

    def __init__(self, config):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        cls = self.__class__
        cls.config_class = config.__class__

        super().__init__(config)

        self.config = config

        # Load values from the configuration
        c = config.to_dict()
        pretrained_model = c["_name_or_path"]
        self.tasks_names = c["puc_tasks"]
        self.tasks_kwargs = c["puc_tasks_kwargs"]

        # Load base model and tasks
        heads, heads_config = cls.get_task_heads(self.tasks_names, self.tasks_kwargs, pretrained_model)
        shared_encoder, task_models_dict = cls.get_base_and_heads(heads, heads_config, config, pretrained_model)
        self.encoder = shared_encoder

        self.update_task_models(task_models_dict)

    def update_task_models(self, task_models_dict):
        self.task_models_dict = task_models_dict
        self.task_models_dict_modules = nn.ModuleDict(task_models_dict)

    @classmethod
    def get_task_model_path(cls, d, task):
        return f"{d}.heads.{task}"

    def from_pretrained_wrapper(self, model_input, device=None):
        cls = self.__class__
        shared_encoder = None

        for task, task_head in self.task_models_dict.items():
            task_path = cls.get_task_model_path(model_input, task)

            if not utils.exists(task_path, f=os.path.isdir):
                raise Exception(f"Provided input model does not exist (task: {task}): '{task_path}'")

            self.task_models_dict[task] = self.task_models_dict[task].from_pretrained(task_path)
            encoder_attr_name = cls.get_encoder_attr_name(self.task_models_dict[task])

            # REMEMBER: I've tried to remove the base model and only store the head relevant content, but when loading the structure is not the same and an exception is raised

            if shared_encoder is None:
                shared_encoder = getattr(self.task_models_dict[task], encoder_attr_name)
                self.encoder = shared_encoder
            else:
                # Replace base model in the head
                setattr(self.task_models_dict[task], encoder_attr_name, shared_encoder)

            self.task_models_dict[task].to(device)

        self.update_task_models(self.task_models_dict)

        two_or_more_tasks = len(self.get_tasks_names()) > 1
        logger.info("Model(s) loaded: %s.heads.%s%s%s", model_input,
                    '{' if two_or_more_tasks else '', ','.join(self.get_tasks_names()),
                    '}' if two_or_more_tasks else '')

    def save_pretrained_wrapper(self, model_output):
        cls = self.__class__

        for task, task_head in self.task_models_dict.items():
            task_path = cls.get_task_model_path(model_output, task)

            # REMEMBER: I've tried to remove the base model and only store the head relevant content, but when loading the structure is not the same and an exception is raised

            self.task_models_dict[task].save_pretrained(task_path)

        two_or_more_tasks = len(self.get_tasks_names()) > 1
        logger.info("Model(s) saved: %s.heads.%s%s%s", model_output,
                    '{' if two_or_more_tasks else '', ','.join(self.get_tasks_names()),
                    '}' if two_or_more_tasks else '')

    @classmethod
    def get_task_heads(cls, tasks, tasks_kwargs, model_source):
        heads = {}
        heads_config = {}

        for task in tasks:
            if task == "urls_classification":
                heads[task] = transformers.AutoModelForSequenceClassification
            elif task == "mlm":
                heads[task] = transformers.AutoModelForMaskedLM
            elif task == "language-detection":
                heads[task] = transformers.AutoModelForSequenceClassification
            else:
                raise Exception(f"Unknown task: {task}")

            task_kwargs = tasks_kwargs[task] if tasks_kwargs and task in tasks_kwargs else {}
            heads_config[task] = transformers.AutoConfig.from_pretrained(model_source, **task_kwargs)

        return heads, heads_config

    def get_base_model(self):
        return self.encoder

    def get_head(self, task):
        return self.task_models_dict_modules[task]

    def get_tasks_names(self):
        return self.tasks_names

    @classmethod
    def get_base_and_heads(cls, heads, heads_config, config, model_name):
        shared_encoder = None
        task_models_dict = {}

        for task_name, model_type in heads.items():
            model = model_type.from_pretrained(model_name, config=heads_config[task_name])
            encoder_attr_name = cls.get_encoder_attr_name(model)

            if shared_encoder is None:
                shared_encoder = getattr(model, encoder_attr_name)
            else:
                # Replace base model in the head
                setattr(model, encoder_attr_name, shared_encoder)

            task_models_dict[task_name] = model

        return shared_encoder, task_models_dict

    @classmethod
    def create(cls, model_name, tasks, tasks_kwargs):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        config = transformers.AutoConfig.from_pretrained(model_name)
        c = config.to_dict()
        c["puc_tasks"] = tasks
        c["puc_tasks_kwargs"] = tasks_kwargs

        config.update(c)

        instance = cls(config)

        return instance

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        # General case
        if model.base_model_prefix:
            return model.base_model_prefix

        # Specific case
        model_class_name = model.__class__.__name__

        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("XLMRoberta"):
            return "roberta"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, *args, **kwargs):
        return self.task_models_dict_modules[task_name](*args, **kwargs)
