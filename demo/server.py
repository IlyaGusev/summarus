# Based on https://github.com/allenai/allennlp-server/blob/master/allennlp_server/commands/server_simple.py

import argparse
import json
import logging
import os

from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_module_and_submodules
from flask import Flask, request, Response, jsonify
from flask import render_template
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

logger = logging.getLogger(__name__)


def make_app(models, predictors, title) -> Flask:
    app = Flask(__name__, template_folder=os.path.dirname(os.path.realpath(__file__)))

    @app.route("/")
    def index() -> Response:
        return render_template("template.html", title=title, models=models, predictors=predictors)

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict() -> Response:
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()
        print(data)

        predictor = Predictor.from_archive(models[data["model"]], data["predictor"])
        prediction = predictor.predict_json(data)
        log_blob = {"inputs": data, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))
        return jsonify(prediction)

    @app.route("/predict_batch", methods=["POST", "OPTIONS"])
    def predict_batch() -> Response:
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()
        predictor = Predictor.from_archive(models[data["model"]], data["predictor"])
        prediction = predictor.predict_batch_json(data)
        return jsonify(prediction)

    return app


def main(model_dir, predictors, cuda_device, title, host, port, include_package):
    for package_name in include_package:
        import_module_and_submodules(package_name)
    check_for_gpu(cuda_device)

    models = dict()
    for file_name in filter(lambda f: f.endswith("tar.gz"), os.listdir(model_dir)):
        full_path = os.path.join(model_dir, file_name)
        model_name = file_name.replace(".tar.gz", "")
        archive = load_archive(
            full_path,
            cuda_device=cuda_device
        )
        models[model_name] = archive

    app = make_app(
        models=models,
        predictors=predictors,
        title=title,
    )
    CORS(app)

    http_server = WSGIServer((host, port), app)
    print(f"Models loaded, serving demo on http://{host}:{port}")
    http_server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="directory with all model archives")
    parser.add_argument('--predictors', type=str, action='append', help='possible predictors',
                        default=["subwords_summary", "words_summary", "subwords_summary_sentences"])
    parser.add_argument("--cuda-device", type=int, default=-1, help="id of GPU to use (if any)")
    parser.add_argument("--title", type=str, help="change the default page title", default="Summarus Demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="interface to serve the demo on")
    parser.add_argument("--port", type=int, default=8000, help="port to serve the demo on")
    parser.add_argument('--include-package', type=str, action='append', default=["summarus"],
                        help='additional packages to include')

    args = parser.parse_args()
    main(**vars(args))
