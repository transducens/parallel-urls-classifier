
class Conf:
    def __init__(self):
        # Configuration provided to flask_server.py (you might need to modify it in order to fit your needs):
        conf = {
            "parallel_likelihood": True,
            "auxiliary_tasks": ["language-identification", "langid-and-urls_classification"],
            "regression": True,
            "target_task": "langid-and-urls_classification",
            "batch_size": 100,
            "force_cpu": False,
            "pretrained_model": "xlm-roberta-base",
            "flask_port": 5000,
            "lowercase": False,
            "streamer_max_latency": 0.1,
            "max_length_tokens": 256,
            "cuda_amp": False,
            "remove_positional_data_from_resource": False,
            "parallel_likelihood": True,
            "url_separator": '/',
            "disable_streamer": False,
            "expect_urls_base64": False,
            "flask_debug": False,
            "remove_authority": False,
            "do_not_run_flask_server": True,
        }

    for k, v in conf.items():
        setattr(self, k, v)

def init(model_input):
    import parallel_urls_classifier.flask_server as flask_server

    conf = Conf()
    conf.model_input = model_input

    flask_server.main(conf)

    return flask_server.app

if __name__ == "__main__":
    init()
