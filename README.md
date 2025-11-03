# Trade Extractor

A Python tool for processing and classifying securities trades from brokerage statements, with support for automated security type classification using open source LLMs and Alpha Vantage API.

## Features

- Extracts trade information from PDF files
- Classifies securities into categories:
  - Stocks
  - ETFs
  - REITs
- Caches classification results to minimize API calls
- Processes trade data and outputs a structured CSV file

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/othonrm/securities-categorizer.git
cd securities-categorizer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your Broker name and your Alpha Vantage API key:
```
BROKER_NAME=""
ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

## Usage

1. Run the trade extractor pointing to your PDF file:
```bash
python trade_extractor.py target.pdf
```

## License

[GNU General Public License v3.0](LICENSE)