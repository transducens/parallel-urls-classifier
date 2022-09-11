
def main():
    import parallel_urls_classifier.parallel_urls_classifier as puc

    puc.cli()

def flask_server():
    import parallel_urls_classifier.flask_server as puc_flask

    puc_flask.cli()
