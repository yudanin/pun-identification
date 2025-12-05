# PIE - Pun Identification Engine

A Python system for identifying and analyzing puns using Claude LLM, FrameNet semantics, and linguistic validation.

## Features

- **Pun Detection**: Identifies puns in natural language text
- **Type Classification**: Categorizes puns as:
  - Homophonic (similar-sounding words)
  - Homographic (same spelling, different meanings)
  - Recursive (self-referential)
  - Antanaclasis (repeated word with different senses)
- **Frame Distance**: Measures semantic distance using FrameNet frames
- **Validation**: Verifies puns using distributional semantics and substitution tests
- **REST API**: Flask-based API for integration

## Installation

```bash
# Clone or copy the project
cd pie

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLP models (optional but recommended)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('framenet_v17')"
```

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your-api-key
```

## Usage

### Command Line

```bash
# Analyze a single sentence
python cli.py "Time flies like an arrow; fruit flies like a banana."

# Analyze from file
python cli.py --file test_sentences.txt

# Interactive mode
python cli.py --interactive

# JSON output
python cli.py --json "I used to be a banker, but I lost interest."
```

### Python API

```python
from pie import PunIdentificationEngine

# Initialize engine
engine = PunIdentificationEngine(api_key="your-api-key")

# Analyze a sentence
result = engine.analyze("Atheism is a non-prophet institution.")

# Check results
if result.has_pun:
    for pun in result.puns:
        print(f"Pun: {pun.word_or_expression}")
        print(f"Type: {pun.pun_type.value}")
        print(f"Senses: {pun.sense1} / {pun.sense2}")
        print(f"Frame Distance: {pun.frame_distance.distance}")
        print(f"Explanation: {pun.explanation}")
```

### REST API

Start the server:
```bash
python api.py
# or for production:
gunicorn api:app
```

Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/status` | Engine status |
| POST | `/configure` | Set API key |
| POST | `/analyze` | Analyze single sentence |
| POST | `/analyze/batch` | Analyze multiple sentences |

Example request:
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sentence": "I used to be a banker, but I lost interest."}'
```

## Response Format

```json
{
  "sentence": "I used to be a banker, but I lost interest.",
  "has_pun": 1,
  "puns": [
    {
      "word_or_expression": "interest",
      "pun_type": "homographic",
      "sense1": "curiosity or attention",
      "sense2": "money paid for borrowing",
      "frame_distance": {
        "distance": 7.0,
        "distance_type": "estimated",
        "explanation": "The frames are in completely different domains...",
        "sense1_frame": {...},
        "sense2_frame": {...}
      },
      "explanation": "The word 'interest' plays on two meanings...",
      "validation": {
        "distributional_valid": true,
        "distributional_explanation": "Both senses are activated...",
        "substitution_valid": true,
        "substitution_explanation": "Both substitutions work...",
        "overall_confidence": 0.9
      },
      "confidence": 0.9
    }
  ],
  "analysis_notes": ""
}
```

## PythonAnywhere Deployment

1. Upload files to PythonAnywhere
2. Create a virtualenv and install requirements
3. Configure WSGI file (update username in `wsgi.py`)
4. Set `ANTHROPIC_API_KEY` in the virtualenv
5. Reload the web app

## Architecture

```
pie/
├── pie/                    # Main package
│   ├── __init__.py        # Package exports
│   ├── engine.py          # Main pun identification engine
│   ├── models.py          # Data models
│   ├── framenet_service.py # FrameNet integration
│   └── validators.py      # Validation logic
├── api.py                 # Flask REST API
├── cli.py                 # Command-line interface
├── wsgi.py                # WSGI entry point
├── requirements.txt       # Dependencies
└── test_sentences.txt     # Example sentences
```

## Pun Types

| Type | Description | Example |
|------|-------------|---------|
| Homophonic | Similar-sounding words | "prophet" / "profit" |
| Homographic | Same spelling, different meaning | "foot" (body part / measurement) |
| Recursive | Self-referential | "Immanuel doesn't pun, he Kant" |
| Antanaclasis | Same word, different senses | "hang together / hang separately" |

## License

MIT License
