import os
import io
import json
import traceback
import pandas as pd
from flask import(Flask, request, jsonify, render_template, redirect, url_for, flash, session)
from flask_cors import CORS
from werkzeug.utils import secure_filename

import  sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.engine import(
    train_session, predict_single, predict_batch, load_session
)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__),'..','static')
)

app.secret_key=os.environ.get('SECRET_KEY', '')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def error_response(msg, code=400):
    if request.is_json or request.headers.get('Accept') == 'application/json':
        return jsonify({'error': msg}), code
    flash(msg,'error')
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return error_response('No file uploaded.')
    file = request.files['file']
    if file.filename == '':
        return error_response('No file selected.')
    if not allowed_file(file.filename):
        return error_response('Only CSV files are supported')
    
    try:
        content = file.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return error_response(f'Could not parse CSV: {str(e)}')
    
    if df.empty or len(df.columns) < 2:
        return error_response('CSV must have at least 2 columns and 1 row')
    
    target_col = request.form.get('target_col', '').strip()
    if not target_col:
        return error_response('Please select a target column.')
    if target_col not in df.columns:
        return error_response(f"Column '{target_col}' not found in CSV.")
    try:
        result = train_session(df, target_col)
    except ValueError as e:
        return error_response(str(e))
    except Exception:
        traceback.print_exc()
        return error_response('Training failed. Check your data and try again.', 500)
    session['current_session_id'] = result['session_id']
    return render_template('results.html', result=result)

@app.route('/predict-form/<session_id>')
def predict_form(session_id):
    try:
        _, _, meta = load_session(session_id)
    except FileNotFoundError:
        flash('Session not found.', 'error')
        return redirect(url_for('index'))
    
    input_data = {}
    for col in meta['feature_cols']:
        val = request.form.get(col, '').strip()
        if col in meta['numeric_cols']:
            try:
                input_data[col] = float(val) if val != '' else None
            except ValueError:
                input_data[col] = None
        else:
            input_data[col] = val if val != '' else None

        
        try:
            result = predict_single(session_id, input_data)
        except Exception as e:
            flash(f'Prediction failed: {str(e)}', 'error')
            return redirect(url_for('predict_form', session_id=session_id))
        
        return render_template('predict.html', meta=meta, session_id=session_id, prediction=result, input_data=input_data)
    
@app.route('/predict-batch/<session_id>', methods=['POST'])
def predict_batch_ui(session_id):
    try:
        _,_,meta = load_session(session_id)
    except FileNotFoundError:
        flash('Session not found.', 'error')
        return redirect(url_for('index'))
    if 'file' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('predict_form', session_id=session_id))
    
    file = request.files['file']
    if not allowed_file(file.filename):
        flash('Only CSV files are supported.','error')
        return redirect(url_for('predict_form' session_id=session_id))
    
    try:
        content = file.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        flash(f'Could not parse CSV: {str(e)}', 'error')
        return redirect(url_for('predict_form', session_id=session_id))
    
    try:
        predictions = predict_batch(session_id, df)
    except Exception as e:
        flash(f'Batch prediction failed: {str(e)}', 'error')
        return redirect(url_for('predict_form', session_id=session_id))
    
    return render_template('predict.html', meta=meta, session_id=session_id,batch_predictions=predictions, batch_count=len(predictions))


@app.route('/api/columns', methods=['POST'])
def get_columns():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}),400
    file = request.files['file']
    try:
        content = file.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content), nrows=5)
        return jsonify({'columns': df.columns.tolist(), 'preview': df.head(3).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def api_train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    target_col = request.form.get('target_col', '')
    if not target_col:
        return jsonify({'error': 'target_col required'}), 400
    try:
        content = file.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content))
        result = train_session(df, target_col)
        result.pop('eda_plots', None)
        result.pop('importance_plot',None)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}),500

@app.route('/api/predict/<session_id>', methods=['POST'])
def api_predict(session_id):
    data = request.get_json
    if not data:
        return jsonify({'error': 'JSON body required'}),400
    try:
        result = predict_single(session_id,data)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}),500


@app.route('/api/predict/<session_id>/batch', methods=['POST'])
def api_predict_batch(session_id):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        content = file.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content))
        predictions = predict_batch(session_id, df)
        return jsonify({'predictions': predictions, 'count':len(predictions)})
    except FileNotFoundError:
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/session/<session_id>', methods=['GET'])
def api_session_info(session_id):
    try:
        _,_, meta = load_session(session_id)
        return jsonify(meta)
    except FileNotFoundError:
        return jsonify({'error': 'Session not found'}), 404

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version':'1.0.0'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)        
    

    
    
