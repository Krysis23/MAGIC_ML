

import io
import json
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.app import app



TITANIC_CSV = """Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,Survived
3,male,22,1,0,7.25,S,0
1,female,38,1,0,71.28,C,1
3,female,26,0,0,7.92,S,1
1,female,35,1,0,53.1,S,1
3,male,35,0,0,8.05,S,0
3,male,,0,0,8.46,Q,0
1,male,54,0,0,51.86,S,0
3,male,2,3,1,21.07,S,0
3,female,27,0,2,11.13,S,1
2,female,14,1,0,30.07,C,1
3,female,4,1,1,16.7,G,1
1,female,58,0,0,26.55,S,1
3,male,20,0,0,8.05,S,0
3,male,39,1,5,31.27,S,0
3,female,14,0,0,7.85,S,0
2,female,55,0,0,16.0,S,1
3,male,2,4,1,29.12,Q,0
2,male,,0,0,13.0,S,1
3,female,31,1,0,18.0,S,0
1,female,35,1,0,83.48,C,1
"""

PREDICT_CSV = """Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
3,male,22,1,0,7.25,S
1,female,38,1,0,71.28,C
"""


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def trained_session(client):
    """Train a model and return the session_id."""
    data = {'file': (io.BytesIO(TITANIC_CSV.encode()), 'titanic.csv'),
            'target_col': 'Survived'}
    rv = client.post('/api/train', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    return rv.get_json()['session_id']



def test_health(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['status'] == 'ok'



def test_get_columns(client):
    data = {'file': (io.BytesIO(TITANIC_CSV.encode()), 'titanic.csv')}
    rv = client.post('/api/columns', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    cols = rv.get_json()['columns']
    assert 'Survived' in cols
    assert 'Age' in cols


def test_get_columns_no_file(client):
    rv = client.post('/api/columns', data={}, content_type='multipart/form-data')
    assert rv.status_code == 400


def test_train_success(client):
    data = {'file': (io.BytesIO(TITANIC_CSV.encode()), 'titanic.csv'),
            'target_col': 'Survived'}
    rv = client.post('/api/train', data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    body = rv.get_json()
    assert 'session_id' in body
    assert 'results' in body
    assert body['problem_type'] in ('binary', 'multiclass', 'regression')


def test_train_no_file(client):
    data = {'target_col': 'Survived'}
    rv = client.post('/api/train', data=data, content_type='multipart/form-data')
    assert rv.status_code == 400


def test_train_bad_target(client):
    data = {'file': (io.BytesIO(TITANIC_CSV.encode()), 'titanic.csv'),
            'target_col': 'NonExistentColumn'}
    rv = client.post('/api/train', data=data, content_type='multipart/form-data')
    assert rv.status_code == 500


def test_predict_single(client, trained_session):
    payload = {'Pclass': 3, 'Sex': 'male', 'Age': 22, 'SibSp': 1,
               'Parch': 0, 'Fare': 7.25, 'Embarked': 'S'}
    rv = client.post(f'/api/predict/{trained_session}',
                     data=json.dumps(payload),
                     content_type='application/json')
    assert rv.status_code == 200
    body = rv.get_json()
    assert 'prediction' in body


def test_predict_unknown_session(client):
    rv = client.post('/api/predict/nonexistent',
                     data=json.dumps({'Age': 22}),
                     content_type='application/json')
    assert rv.status_code == 404


def test_predict_no_body(client, trained_session):
    rv = client.post(f'/api/predict/{trained_session}',
                     content_type='application/json')
    assert rv.status_code == 400


def test_predict_batch(client, trained_session):
    data = {'file': (io.BytesIO(PREDICT_CSV.encode()), 'predict.csv')}
    rv = client.post(f'/api/predict/{trained_session}/batch',
                     data=data, content_type='multipart/form-data')
    assert rv.status_code == 200
    body = rv.get_json()
    assert 'predictions' in body
    assert body['count'] == 2


def test_predict_batch_no_file(client, trained_session):
    rv = client.post(f'/api/predict/{trained_session}/batch',
                     data={}, content_type='multipart/form-data')
    assert rv.status_code == 400



def test_session_info(client, trained_session):
    rv = client.get(f'/api/session/{trained_session}')
    assert rv.status_code == 200
    body = rv.get_json()
    assert body['target_col'] == 'Survived'
    assert 'feature_cols' in body


def test_session_not_found(client):
    rv = client.get('/api/session/does-not-exist')
    assert rv.status_code == 404



def test_home_page(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'AutoML' in rv.data