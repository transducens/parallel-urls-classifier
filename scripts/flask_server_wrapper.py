
def init(model_input, batch_size=16, streamer_max_latency=0.1, target_task="urls_classification"):
    import sys
    import parallel_urls_classifier.flask_server as flask_server

    sys.argv = [sys.argv[0]] # Remove all provided args

    # Inject args that will be used by the Flask server
    sys.argv.extend([
        "--batch-size", str(batch_size),
        "--parallel-likelihood",
        "--auxiliary-tasks", "language-identification", "langid-and-urls_classification",
        "--target-task", target_task,
        "--regression",
        "--streamer-max-latency", str(streamer_max_latency),
        "--do-not-run-flask-server", # Necessary for gunicorn in order to work properly
        "--expect-urls-base64",
        #"--verbose",
        #"--disable-streamer",
        model_input
    ])

    flask_server.cli()

    return flask_server.app
