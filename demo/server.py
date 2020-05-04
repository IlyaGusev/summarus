# Based on https://github.com/allenai/allennlp-server/blob/master/allennlp_server/commands/server_simple.py

import argparse
import json
import logging
import os
import sys
from string import Template
from typing import List, Callable

from allennlp.common import JsonDict
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules
from flask import Flask, request, Response, jsonify, send_file, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

logger = logging.getLogger(__name__)


def make_app(
    predictor: Predictor,
    field_names: List[str] = None,
    sanitizer: Callable[[JsonDict], JsonDict] = None,
    title: str = "Summarus Demo",
) -> Flask:
    assert field_names is not None, "No field_names passed."

    app = Flask(__name__)

    @app.route("/")
    def index() -> Response:
        html = _html(title, field_names)
        return Response(response=html, status=200)

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict() -> Response:
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_json(data)
        if sanitizer is not None:
            prediction = sanitizer(prediction)

        log_blob = {"inputs": data, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify(prediction)

    @app.route("/predict_batch", methods=["POST", "OPTIONS"])
    def predict_batch() -> Response:
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_batch_json(data)
        if sanitizer is not None:
            prediction = [sanitizer(p) for p in prediction]

        return jsonify(prediction)

    return app


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
        args.archive_path,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )

    return Predictor.from_archive(archive, args.predictor)


_SINGLE_INPUT_TEMPLATE = Template(
    """
        <div class="form__field">
            <label for="input-$field_name">$field_name</label>
            <textarea type="text" id="input-$field_name" type="text" required value placeholder="input goes here"></textarea>
        </div>
"""
)


def _html(title: str, field_names: List[str]) -> str:
    """
    Returns bare bones HTML for serving up an input form with the
    specified fields that can render predictions from the configured model.
    """
    inputs = "".join(
        _SINGLE_INPUT_TEMPLATE.substitute(field_name=field_name) for field_name in field_names
    )

    script_path = os.path.dirname(os.path.realpath(__file__))
    quoted_field_names = (f"'{field_name}'" for field_name in field_names)
    quoted_field_list = f"[{','.join(quoted_field_names)}]"
    with open(os.path.join(script_path, "template.html"), "r") as r:
        page_template = Template(r.read())
    with open(os.path.join(script_path, "style.css"), "r") as r:
        css = r.read()

    return page_template.substitute(title=title, css=css, inputs=inputs, qfl=quoted_field_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-path", type=str, required=True, help="path to trained archive file")
    parser.add_argument("--predictor", type=str, required=True, help="name of predictor")
    parser.add_argument("--weights-file", type=str, help="a path that overrides which weights file to use")
    parser.add_argument("--cuda-device", type=int, default=-1, help="id of GPU to use (if any)")
    parser.add_argument("-o", "--overrides", type=str, default="",
                        help="a JSON structure used to override the experiment configuration")
    parser.add_argument("--title", type=str, help="change the default page title", default="Summarus Demo")
    parser.add_argument(
        "--field-name",
        type=str,
        action="append",
        dest="field_names",
        metavar="FIELD_NAME",
        help="field names to include in the demo",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="interface to serve the demo on")
    parser.add_argument("-p", "--port", type=int, default=8000, help="port to serve the demo on")
    parser.add_argument('--include-package', type=str, action='append', default=[], help='additional packages to include')

    args = parser.parse_args()
    for package_name in args.include_package:
        import_submodules(package_name)
    predictor = _get_predictor(args)

    app = make_app(
        predictor=predictor,
        field_names=args.field_names,
        title=args.title,
    )
    CORS(app)

    http_server = WSGIServer((args.host, args.port), app)
    print(f"Model loaded, serving demo on http://{args.host}:{args.port}")
    http_server.serve_forever()
