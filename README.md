# Parallel URLs Classifier

`parallel-urls-classifier` (PUC) is a tool implemented in Python that allows to infer whether a pair of URLs link to parallel documents (i.e., documents with the same content but written in different languages). You can either get a textual description `positive`/`negative` or the probability that the URL pair links to parallel documents.

The code provided in this repo allows to train new models. If you want to use the released models, see the HuggingFace page (there is a usage example): https://huggingface.co/Transducens/xlm-roberta-base-parallel-urls-classifier. The released models in HuggingFace are not directly compatible with this code since it contains code ported from HuggingFace to implement multitasking, but if multitasking was not used, models can be manually converted to/from the HuggingFace version. This code should be used if you plan to train new models.

## Installation

To install PUC first clone the code from the repository:

```bash
git clone https://github.com/transducens/parallel-urls-classifier.git
```

Optionally, create a conda environment to isolate the python dependencies:

```bash
conda create -n PUC -c conda-force python==3.8.5
conda activate PUC
```

Install PUC:

```bash
cd parallel-urls-classifier

pip3 install .
```

Check out the installation:

```bash
parallel-urls-classifier --help
```

## Usage

```
usage: parallel-urls-classifier [-h] [--batch-size BATCH_SIZE]
                                [--block-size BLOCK_SIZE]
                                [--max-tokens MAX_TOKENS] [--epochs EPOCHS]
                                [--do-not-fine-tune] [--freeze-whole-model]
                                [--dataset-workers DATASET_WORKERS]
                                [--pretrained-model PRETRAINED_MODEL]
                                [--max-length-tokens MAX_LENGTH_TOKENS]
                                [--model-input MODEL_INPUT]
                                [--model-output MODEL_OUTPUT] [--inference]
                                [--inference-from-stdin]
                                [--inference-lang-using-url2lang]
                                [--parallel-likelihood]
                                [--threshold THRESHOLD]
                                [--imbalanced-strategy {none,over-sampling,weighted-loss}]
                                [--patience PATIENCE] [--train-until-patience]
                                [--do-not-load-best-model]
                                [--overwrite-output-model]
                                [--remove-authority]
                                [--remove-positional-data-from-resource]
                                [--add-symmetric-samples] [--force-cpu]
                                [--log-directory LOG_DIRECTORY] [--regression]
                                [--url-separator URL_SEPARATOR]
                                [--url-separator-new-token]
                                [--learning-rate LEARNING_RATE]
                                [--optimizer {none,adam,adamw,sgd}]
                                [--optimizer-args beta1 beta2 eps weight_decay]
                                [--lr-scheduler {none,linear,CLR,inverse_sqrt}]
                                [--lr-scheduler-args warmup_steps]
                                [--re-initialize-last-n-layers RE_INITIALIZE_LAST_N_LAYERS]
                                [--cuda-amp] [--llrd]
                                [--stringify-instead-of-tokenization]
                                [--lowercase]
                                [--auxiliary-tasks [{mlm,language-identification,langid-and-urls_classification} [{mlm,language-identification,langid-and-urls_classification} ...]]]
                                [--auxiliary-tasks-weights [AUXILIARY_TASKS_WEIGHTS [AUXILIARY_TASKS_WEIGHTS ...]]]
                                [--freeze-embeddings-layer]
                                [--remove-instead-of-truncate]
                                [--best-dev-metric {loss,Macro-F1,MCC}]
                                [--task-dev-metric {urls_classification,language-identification,langid-and-urls_classification}]
                                [--auxiliary-tasks-flags [{language-identification_add-solo-urls-too,language-identification_target-applies-only-to-trg-side,langid-and-urls_classification_reward-if-only-langid-is-correct-too} [{language-identification_add-solo-urls-too,language-identification_target-applies-only-to-trg-side,langid-and-urls_classification_reward-if-only-langid-is-correct-too} ...]]]
                                [--do-not-train-main-task] [--pre-load-shards]
                                [--seed SEED] [--plot] [--plot-path PLOT_PATH]
                                [--lock-file LOCK_FILE]
                                [--waiting-time WAITING_TIME] [-v]
                                dataset_train_filename dataset_dev_filename
                                dataset_test_filename

Parallel URLs classifier

positional arguments:
  dataset_train_filename
                        Filename with train data (TSV format). You can provide
                        multiple files separated using ':' and each of them
                        will be used one for each epoch using round-robin
                        strategy
  dataset_dev_filename  Filename with dev data (TSV format)
  dataset_test_filename
                        Filename with test data (TSV format)

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size. Elements which will be processed before
                        proceed to train, but the whole batch will be
                        processed in blocks in order to avoid OOM errors
                        (default: 16)
  --block-size BLOCK_SIZE
                        Block size. Elements which will be provided to the
                        model at once (default: None)
  --max-tokens MAX_TOKENS
                        Process batches in groups tokens size (fairseq style).
                        Batch size is still relevant since the value is used
                        when batches are needed (e.g. sampler from dataset)
                        (default: -1)
  --epochs EPOCHS       Epochs (default: 3)
  --do-not-fine-tune    Do not apply fine-tuning to the base model (default
                        weights) (default: False)
  --freeze-whole-model  Do not apply fine-tuning to the whole model, not only
                        the base model (default: False)
  --dataset-workers DATASET_WORKERS
                        No. workers when loading the data in the dataset. When
                        negative, all available CPUs will be used (default:
                        -1)
  --pretrained-model PRETRAINED_MODEL
                        Pretrained model (default: xlm-roberta-base)
  --max-length-tokens MAX_LENGTH_TOKENS
                        Max. length for the generated tokens (default: 256)
  --model-input MODEL_INPUT
                        Model input path which will be loaded (default: None)
  --model-output MODEL_OUTPUT
                        Model output path where the model will be stored
                        (default: None)
  --inference           Do not train, just apply inference (flag --model-input
                        is recommended). If this option is set, it will not be
                        necessary to provide the input dataset (default:
                        False)
  --inference-from-stdin
                        Read inference from stdin (default: False)
  --inference-lang-using-url2lang
                        When --inference is provided, if the language is
                        necessary, url2lang will be used. The langs will be
                        provided anyway to the model if needed, but the result
                        will be ignored. The results of the language tasks
                        will be either 1 or 0 if the lang matchs (default:
                        False)
  --parallel-likelihood
                        Print parallel likelihood instead of classification
                        string (inference) (default: False)
  --threshold THRESHOLD
                        Only print URLs which have a parallel likelihood
                        greater than the provided threshold (inference)
                        (default: -inf)
  --imbalanced-strategy {none,over-sampling,weighted-loss}
                        Strategy for dealing with imbalanced data (default:
                        none)
  --patience PATIENCE   Patience before stopping the training (default: 0)
  --train-until-patience
                        Train until patience value is reached (--epochs will
                        be ignored in order to stop, but will still be used
                        for other actions like LR scheduler) (default: False)
  --do-not-load-best-model
                        Do not load best model for final dev and test
                        evaluation (--model-output is necessary) (default:
                        False)
  --overwrite-output-model
                        Overwrite output model if it exists (initial loading)
                        (default: False)
  --remove-authority    Remove protocol and authority from provided URLs
                        (default: False)
  --remove-positional-data-from-resource
                        Remove content after '#' in the resorce (e.g.
                        https://www.example.com/resource#position ->
                        https://www.example.com/resource) (default: False)
  --add-symmetric-samples
                        Add symmetric samples for training (if (src, trg) URL
                        pair is provided, (trg, src) URL pair will be provided
                        as well) (default: False)
  --force-cpu           Run on CPU (i.e. do not check if GPU is possible)
                        (default: False)
  --log-directory LOG_DIRECTORY
                        Directory where different log files will be stored
                        (default: None)
  --regression          Apply regression instead of binary classification
                        (default: False)
  --url-separator URL_SEPARATOR
                        Separator to use when URLs are stringified (default:
                        /)
  --url-separator-new-token
                        Add special token for URL separator (default: False)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 1e-05)
  --optimizer {none,adam,adamw,sgd}
                        Optimizer (default: adamw)
  --optimizer-args beta1 beta2 eps weight_decay
                        Args. for the optimizer (in order to see the specific
                        configuration for a optimizer, use -h and set
                        --optimizer) (default: (0.9, 0.999, 1e-08, 0.01))
  --lr-scheduler {none,linear,CLR,inverse_sqrt}
                        LR scheduler (default: inverse_sqrt)
  --lr-scheduler-args warmup_steps
                        Args. for LR scheduler (in order to see the specific
                        configuration for a LR scheduler, use -h and set --lr-
                        scheduler) (default: ('10%',))
  --re-initialize-last-n-layers RE_INITIALIZE_LAST_N_LAYERS
                        Re-initialize last N layers from pretained model (will
                        be applied only when fine-tuning the model) (default:
                        1)
  --cuda-amp            Use CUDA AMP (Automatic Mixed Precision) (default:
                        False)
  --llrd                Apply LLRD (Layer-wise Learning Rate Decay) (default:
                        False)
  --stringify-instead-of-tokenization
                        Preprocess URLs applying custom stringify instead of
                        tokenization (default: False)
  --lowercase           Lowercase URLs while preprocessing (default: False)
  --auxiliary-tasks [{mlm,language-identification,langid-and-urls_classification} [{mlm,language-identification,langid-and-urls_classification} ...]]
                        Tasks which will try to help to the main task
                        (multitasking) (default: None)
  --auxiliary-tasks-weights [AUXILIARY_TASKS_WEIGHTS [AUXILIARY_TASKS_WEIGHTS ...]]
                        Weights for the loss of the auxiliary tasks. If none
                        is provided, the weights will be 1, but if any is
                        provided, as many weights as auxiliary tasks will have
                        to be provided (default: None)
  --freeze-embeddings-layer
                        Freeze embeddings layer (default: False)
  --remove-instead-of-truncate
                        Remove pairs of URLs which would need to be truncated
                        (if not enabled, truncation will be applied). This
                        option will be only applied to the training set
                        (default: False)
  --best-dev-metric {loss,Macro-F1,MCC}
                        Which metric should be maximized or minimized when dev
                        is being evaluated in order to save the best model
                        (default: Macro-F1)
  --task-dev-metric {urls_classification,language-identification,langid-and-urls_classification}
                        Task which will be used in order to save the best
                        model. It will also be used in order to replace the
                        main task if --do-not-train-main-task is set (default:
                        urls_classification)
  --auxiliary-tasks-flags [{language-identification_add-solo-urls-too,language-identification_target-applies-only-to-trg-side,langid-and-urls_classification_reward-if-only-langid-is-correct-too} [{language-identification_add-solo-urls-too,language-identification_target-applies-only-to-trg-side,langid-and-urls_classification_reward-if-only-langid-is-correct-too} ...]]
                        Set of options which will set up some aspects of the
                        auxiliary tasks (default: None)
  --do-not-train-main-task
                        Main task (URLs classification) will not be trained.
                        Auxiliary task will be needed (default: False)
  --pre-load-shards     Load all shards at beginning one by one in order to
                        get some statistics needed for some features. This
                        option is optional, but if not set, some features
                        might not work as expected (e.g. linear LR scheduler)
                        (default: False)
  --seed SEED           Seed in order to have deterministic results (not fully
                        guaranteed). Set a negative number in order to disable
                        this feature (default: 71213)
  --plot                Plot statistics (matplotlib pyplot) in real time
                        (default: False)
  --plot-path PLOT_PATH
                        If set, the plot will be stored instead of displayed
                        (default: None)
  --lock-file LOCK_FILE
                        If set, and the file does not exist, it will be
                        created once the training finishes. If does exist, the
                        training will not be executed (default: None)
  --waiting-time WAITING_TIME
                        Waiting time, if needed for letting the user react
                        (default: 20)
  -v, --verbose         Verbose logging mode (default: False)
```

## CLI examples

Train a new model:

```bash
parallel-urls-classifier /path/to/datasets/{train,dev,test}.tsv \
  --regression --epochs 10 --patience 5 --batch-size 140 --train-until-patience \
  --model-output /tmp/PUC_model
```

Interactive inference:

```bash
parallel-urls-classifier --regression --parallel-likelihood --inference \
  --model-input /tmp/PUC_model
```

Inference using data from a file:

```bash
cat /path/to/datasets/test.tsv \
  | parallel-urls-classifier --regression --parallel-likelihood --inference --inference-from-stdin \
      --model-input /tmp/PUC_model
```

## Inference using Gunicorn

You can use different nodes to perform inference and to execute the model. For more information to execute the model, see:

```bash
parallel-urls-classifier-server --help
```

You may also want to look at the file `scripts/init_flask_server_with_gunicorn.sh` for a specific example to start the server.

The node performing the inference provides information to the node running the model. The information is provided through HTTP requests. Example if we run the model on `127.0.0.1`:

```bash
curl http://127.0.0.1:5000/inference -X POST \
  -d "src_urls=https://domain/resource1&trg_urls=https://domain/resource2"
```

## Acknowledgements

Work co-funded by the Conselleria de Innovación, Universidades, Ciencia y Sociedad Digital (Generalitat Valenciana) and the European Social Fund through grant CIACIF/2021/365, and by the Spanish Ministry of Science and Innovation (MCIN), the Spanish Research Agency (AEI/10.13039/501100011033) and the European Regional Development Fund A way to make Europe through project grant PID2021-127999NB-I00.

![European Social Fund](img/logo-FSE.jpg)
