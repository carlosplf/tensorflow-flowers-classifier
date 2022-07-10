from flask import Blueprint
from ml_runner import runner

ml_router = Blueprint("ml_router", __name__)


@ml_router.route('/ml_test')
def test_ml_router():
    response = {"Foo": "Bar"}
    return response


@ml_router.route('/ml_train/<n_epochs>')
def train_model(n_epochs):
    # Call the ML Runner training routine. Return to the caller the
    # history object.
    data = {"history": runner.run_training(int(n_epochs))}
    return data

