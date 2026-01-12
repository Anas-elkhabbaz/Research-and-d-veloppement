"""
Web interface for SM-UMT Translation.
Flask-based web application with progress tracking.
"""

from flask import Flask, render_template, request, jsonify, Response
import os
import sys
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sm_umt.config import Config
from sm_umt.translator import SMUMTTranslator
from sm_umt.evaluation import get_sample_data
from sm_umt.utils import get_language_name

app = Flask(__name__)

# Store for progress updates
progress_store = {"status": "ready", "message": "", "progress": 0}


def update_progress(status, message, progress):
    """Update progress store."""
    global progress_store
    progress_store = {"status": status, "message": message, "progress": progress}
    print(f"[PROGRESS] {progress}% - {message}")


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation request."""
    global progress_store
    
    try:
        data = request.get_json()
        print(f"\n{'='*50}")
        print("Received translation request")
        print(f"Data: {data}")
        print(f"{'='*50}\n")
        
        text = data.get('text', '').strip()
        src_lang = data.get('src_lang', 'fra')
        tgt_lang = data.get('tgt_lang', 'eng')
        api_key = data.get('api_key', os.environ.get('GEMINI_API_KEY', ''))
        
        if not text:
            return jsonify({'error': 'Please enter text to translate'}), 400
        
        if not api_key:
            return jsonify({'error': 'Please provide a Gemini API key'}), 400
        
        update_progress("working", "Initializing translator...", 10)
        
        # Create translator
        config = Config(src_lang=src_lang, tgt_lang=tgt_lang)
        translator = SMUMTTranslator(config, api_key=api_key)
        
        update_progress("working", "Mining word translations...", 25)
        
        # Get sample data for ICL mining
        sample_data = get_sample_data(src_lang, tgt_lang, 10)
        source_samples = [s for s, t in sample_data]
        
        # Run word mining
        translator.mine_word_translations(source_samples)
        
        update_progress("working", "Creating synthetic parallel data...", 50)
        translator.create_synthetic_parallel(source_samples)
        
        update_progress("working", "Translating sentence...", 75)
        
        # Translate
        translation = translator.translate_sentence(text)
        
        update_progress("done", "Translation complete!", 100)
        
        print(f"\n{'='*50}")
        print(f"Translation result: {translation}")
        print(f"{'='*50}\n")
        
        return jsonify({
            'success': True,
            'source': text,
            'translation': translation,
            'src_lang': get_language_name(src_lang),
            'tgt_lang': get_language_name(tgt_lang)
        })
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}\n")
        
        update_progress("error", str(e), 0)
        return jsonify({'error': str(e)}), 500


@app.route('/progress')
def get_progress():
    """Get current progress."""
    return jsonify(progress_store)


@app.route('/languages')
def get_languages():
    """Get available languages."""
    languages = [
        {'code': 'fra', 'name': 'French'},
        {'code': 'eng', 'name': 'English'},
        {'code': 'arb', 'name': 'Arabic'}
    ]
    return jsonify(languages)


@app.route('/simple_translate', methods=['POST'])
def simple_translate():
    """Simple translation using just the LLM (faster for testing)."""
    try:
        data = request.get_json()
        print(f"\n[SIMPLE] Received: {data}\n")
        
        text = data.get('text', '').strip()
        src_lang = data.get('src_lang', 'fra')
        tgt_lang = data.get('tgt_lang', 'eng')
        api_key = data.get('api_key', '')
        
        if not text or not api_key:
            return jsonify({'error': 'Missing text or API key'}), 400
        
        # Direct LLM translation
        from sm_umt.llm_client import LLMClient
        from sm_umt.prompts import create_sentence_translation_prompt
        
        client = LLMClient(api_key=api_key)
        prompt = create_sentence_translation_prompt(
            text, 
            get_language_name(src_lang), 
            get_language_name(tgt_lang)
        )
        
        translation = client.generate(prompt)
        
        print(f"[SIMPLE] Result: {translation}\n")
        
        return jsonify({
            'success': True,
            'source': text,
            'translation': translation,
            'src_lang': get_language_name(src_lang),
            'tgt_lang': get_language_name(tgt_lang)
        })
        
    except Exception as e:
        print(f"[SIMPLE] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("SM-UMT Web Interface")
    print("="*50)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
