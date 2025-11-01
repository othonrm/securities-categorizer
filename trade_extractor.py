import pdfplumber
import pandas as pd
import re
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os
import json
import warnings
from dotenv import load_dotenv

load_dotenv()

categoriesEnum = [
    "Stock",
    "ETF's Internacionais",
    "Reits",
]


class SecurityClassifier:
    def __init__(self, alpha_vantage_key: str = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv(
            'ALPHA_VANTAGE_API_KEY')
        self.cache = {}

        # Limit because the Alpha Vantage free tier only supports 5 calls per minute
        self.rate_limit_delay = 12

    def classify_symbols_bulk(self, symbols: List[str], matches) -> Dict[str, str]:
        results = {}

        # Check cache first
        uncached_symbols = []
        for symbol in symbols:
            if symbol in self.cache:
                results[symbol] = self.cache[symbol]
            else:
                uncached_symbols.append(symbol)

        if not uncached_symbols:
            return results

        # For each uncached symbol, get its match data
        uncached_matches = []
        for symbol in uncached_symbols:
            symbolData = next(
                (m for m in matches if m[0] == symbol), None)
            if symbolData:
                uncached_matches.append(symbolData)

        print(
            f"\nFound {len(uncached_matches)} uncached matches data to process using AI")

        # self._update_cache_with_ai_classification(uncached_matches)

        # Redo uncached symbols after AI classification
        uncached_symbols = []
        for symbol in symbols:
            if symbol in self.cache:
                results[symbol] = self.cache[symbol]
            else:
                uncached_symbols.append(symbol)

        print(
            f"\nClassifying {len(uncached_symbols)} symbols via API: {', '.join(uncached_symbols)}")

        index = 0

        # Process uncached symbols
        for symbol in uncached_symbols:
            index = index + 1

            hasCalledAPI = False

            # Find the match data for the current symbol
            symbolData = next(
                (m for m in matches if m[0] == symbol), None)

            # print(f"Classifying symbol with data: {symbol} - {symbolData}")

            if 'ETF' in symbolData[1]:
                print(f"Classified {symbol} as ETF based on company name")
                classification = "ETF's Internacionais"
            elif 'REIT' in symbolData[1]:
                print(f"Classified {symbol} as REIT based on company name")
                classification = 'Reits'
            else:
                classification = self._classify_with_symbol_search(symbol)

                if not classification:
                    print(
                        f"Falling back to heuristics for symbol: {symbol}")
                    classification = self._classify_with_heuristics(symbol)
                else:
                    hasCalledAPI = True

            results[symbol] = classification
            self.cache[symbol] = classification

            # Rate limiting between API calls
            # Don't delay after last call
            if symbol != uncached_symbols[-1] and hasCalledAPI:
                print(
                    f"Waiting {self.rate_limit_delay}s till next call ({index + 1}/{len(uncached_symbols)})")
                time.sleep(self.rate_limit_delay)

        return results

    def classify_symbol(self, symbol: str, matches) -> str:
        return self.classify_symbols_bulk([symbol], matches)[symbol]

    def _update_cache_with_ai_classification(self, symbols_data) -> Dict[str, str]:
        models = [
            "gpt-oss:20b",
            "gemma3",
            "gemma3:12b",
        ]

        try:
            url = "http://localhost:11434/api/chat"
            params = {
                "model": models[2],
                "format": "json",
                "think": False,
                "stream": False,
                "keep_alive": "5m",
                "messages": [
                    {
                        "role": "user",
                        "content": """Return just a json with the following pattern:
                        { Symbol: “category” }
                        Using the following options:
                        - 'ETF's Internacionais’ for etfs
                        - ‘Reits' for Reits
                        - 'Stock' for stocks
                        For any given symbol/company/stock/etf that you don't have the data or are not 100% sure, use “NO_MATCH” as the category.
                        Classify the following symbols and companies:
                        """ + json.dumps(symbols_data)
                    }
                ],
            }

            aiTimeoutSeconds = 120

            print(
                f"\nSending request to AI classifier (local Ollama API) with a timeout of {aiTimeoutSeconds} seconds...")

            try:
                response = requests.post(
                    url, json=params, timeout=aiTimeoutSeconds)
                data = response.json()

                try:
                    importantPart = data.get('message').get('content')
                    data = json.loads(importantPart)

                except Exception as e:
                    print(f"Error parsing AI response content: {e}")
                    return None

            except Exception as e:
                print(f"Error calling AI classifier: {e}")
                return None

            print(f"\nParsed AI response content: {data}")

            for symbol_entry in symbols_data:
                symbol = symbol_entry[0]
                classification = data.get(symbol, "NO_MATCH")
                if classification == "NO_MATCH" or classification not in categoriesEnum:
                    print(
                        f"AI could not classify {symbol}, got {classification}.")
                else:
                    self.cache[symbol] = classification

            print(
                f"\nAI classification complete! {len(data)} symbols classified.")

            return

        except Exception as e:
            print(
                f"Error with AI Search API for {len(symbols_data)} symbols: {e}")
            return None

    def _classify_with_symbol_search(self, symbol: str) -> Optional[str]:
        if not self.alpha_vantage_key:
            print(
                "\n ⚠️  Alpha Vantage API key not set, skipping Symbol Search API ⚠️ \n")
            return None

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': symbol,
                'apikey': self.alpha_vantage_key
            }

            print(f"sending request to {url} for {symbol}")

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if 'bestMatches' in data and data['bestMatches']:
                # Look for exact symbol match
                for match in data['bestMatches']:
                    if match.get('1. symbol', '').upper() == symbol.upper():
                        match_type = match.get('3. type', '').upper()
                        name = match.get('2. name', '').upper()

                        # Classify based on type and name
                        if 'ETF' in match_type or 'ETF' in name:
                            return "ETF's Internacionais"
                        elif 'REIT' in match_type or 'REIT' in name or 'REAL ESTATE' in name:
                            return 'Reits'
                        elif 'EQUITY' in match_type or 'COMMON STOCK' in match_type:
                            return 'Stock'
                        else:
                            print(f"Unknown type for {symbol}: {match_type}")
                            return ''  # Default blank

            print(f"No matches found for {symbol} in Symbol Search API")
            return None

        except Exception as e:
            print(
                f"Error with Alpha Vantage Symbol Search API for {symbol}: {e}")
            return None

    def _classify_with_heuristics(self, symbol: str) -> str:
        # Known ETF patterns
        etf_patterns = [
            r'^SPY$', r'^QQQ$', r'^IWM$', r'^VTI$', r'^VOO$',
            r'^TLT$', r'^GLD$', r'^SLV$', r'^XL[A-Z]$',
            r'^I[A-Z]{2}$', r'^V[A-Z]{2}$', r'^XAR$', r'^XME$',
            r'.*ETF$'
        ]

        # Known REIT patterns
        reit_patterns = [
            r'^[A-Z]*REIT$', r'^REI[A-Z]$', r'^VNQ$', r'^SCHH$'
        ]

        # Check ETF patterns
        for pattern in etf_patterns:
            if re.match(pattern, symbol):
                return 'ETF'

        # Check REIT patterns
        for pattern in reit_patterns:
            if re.match(pattern, symbol):
                return 'REIT'

        # Default to Stock
        return 'Stock'


class AdvancedTradeExtractor():
    def __init__(self, alpha_vantage_key: str):
        self.trade_data = []
        self.classifier = SecurityClassifier(alpha_vantage_key)

    def _parse_trades(self, text: str) -> List[Dict]:
        trades = []

        # Extract confirmation date
        confirmation_date = self._extract_confirmation_date(text)

        # Use regex to find trade patterns more precisely
        # Pattern for: SYMBOL COMPANY_NAME ACTION TIME QUANTITY PRICE DATE DATE
        # Old
        # trade_pattern = r'([A-Z]{1,5})\s+([A-Z\s&\.\-]+?)\s+([A-Z]+)\s+Buy\s+([\d:]+\s+[AP]M)\s+([\d\.]+)\s+([\d\.]+)\s+(\d{1,2}/\d{1,2}/\d{4})'
        # New regex to avoid matching companiy names like "VANGUARD S&P 500 ETF INC" or "ISHARES TR 0-3 MNTH TREASRY"
        # Before update to support sell and negative rows
        # trade_pattern = r'^([A-Z]{1,5})\s+([A-Z\s&\.\-0-9]+?)\s+([A-Z]+)\s+Buy\s+([\d:]+\s+[AP]M)\s+([\d\.]+)\s+([\d\.]+)\s+(\d{1,2}/\d{1,2}/\d{4})'
        trade_pattern = r'^([A-Z]{1,5})\s+([A-Z\s&\.\-0-9]+?)\s+([A-Z]+)\s+(Buy|Sell)\s+([\d:]+\s+[AP]M)\s+(\-?[\d\.]+)\s+([\d\.]+)\s+(\d{1,2}/\d{1,2}/\d{4})'

        matches = re.findall(trade_pattern, text, re.MULTILINE)

        if not matches:
            print(
                "No trade matches found with advanced regex, falling back to basic parser")
            # Fallback to parent method if regex fails
            return super()._parse_trades(text)

        # Extract symbols for bulk classification
        symbols = [match[0] for match in matches]

        if symbols:
            print(
                f"\nFound {len(symbols)} symbols to classify: {', '.join(set(symbols))}")
            symbol_classifications = self.classifier.classify_symbols_bulk(
                list(set(symbols)), matches)
            print("\nClassification complete!")

        for match in matches:
            try:
                symbol, company, security_type, action, time, quantity, price, trade_date = match
            except ValueError as e:
                print(f"Error unpacking trade match {match}: {e}")
                continue

            # Get classification from bulk results
            category = symbol_classifications.get(symbol, 'Stock')

            action = 'C' if 'Buy' in action else 'V'

            if action == 'V':
                quantity = quantity.replace('-', '')

            trade = {
                'Data operação': self._format_date(confirmation_date or trade_date),
                'Categoria': category,
                'Código Ativo': symbol,
                'Operação C/V': action,
                'Quantidade': self._format_number(float(quantity)),
                'Preço unitário': self._format_number(float(price)),
                'Corretora': os.getenv('BROKER_NAME', 'Unknown'),
            }

            trades.append(trade)

        return trades

    def _extract_confirmation_date(self, text: str) -> Optional[str]:
        # Look for "Confirmation Date : MM/DD/YYYY" pattern
        date_match = re.search(
            r'Confirmation Date\s*:\s*(\d{1,2}/\d{1,2}/\d{4})', text)
        if date_match:
            return date_match.group(1)

        # Look for "Trade Date" in the data
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        if date_match:
            return date_match.group(1)

        return None

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:

        # Not working
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="CropBox missing from /Page, defaulting to MediaBox")

        warnings.simplefilter("ignore")
        try:
            warnings.simplefilter("ignore")
            with pdfplumber.open(pdf_path, strict_metadata=True) as pdf:
                all_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"

                # Extract trades from the combined text
                trades = self._parse_trades(all_text)
                return trades

        except Exception as e:
            print(f"Error reading PDF: {e}")
            return []

    def process_pdf(self, pdf_path: str, output_path: str = None):
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return

        print(f"Starting PDF processing: {pdf_path}")
        print("\n---------------------------------")
        trades = self.extract_from_pdf(pdf_path)

        if not trades:
            print("No trades found in PDF")
            return

        # Print summary
        print(f"\nFound {len(trades)} trades:")
        # only print trades if --debug is enabled
        if '--debug' in sys.argv:
            for trade in trades:
                print(
                    f"  {trade['Código Ativo']} ({trade['Categoria']}): {trade['Quantidade']} @ {trade['Preço unitário']}")

        # Save to CSV
        if output_path is None:
            output_path = pdf_path.replace('.pdf', '_trades.csv')

        self.save_to_csv(trades, output_path)
        return trades

    def save_to_csv(self, trades: List[Dict], output_path: str):
        if not trades:
            print("No trades to save")
            return

        df = pd.DataFrame(trades)

        # Save with semicolon separator to handle comma decimal separator
        df.to_csv(output_path, index=False, sep=';', decimal=',')
        print(f"Saved {len(trades)} trades to {output_path}")

    def _format_number(self, number: float) -> str:
        return f"{number:.8f}".rstrip('0').rstrip('.').replace('.', ',')

    def _format_date(self, date_str: str) -> str:
        if not date_str:
            return ""

        try:
            # Parse MM/DD/YYYY
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
            # Format to DD/M/YY
            return date_obj.strftime('%-d/%-m/%y')
        except:
            return date_str


def main():
    if len(sys.argv) < 2:
        print("Usage: python trade_extractor.py <pdf_file> [output_csv]")
        print("Example: python trade_extractor.py trade_confirmation.pdf trades.csv")
        print("\nNote: Set ALPHA_VANTAGE_API_KEY environment variable for symbol classification")
        return

    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(
        sys.argv) > 2 and sys.argv[2] != "--debug" else None

    print("=======================================")
    print("🚀 Starting Advanced Trade Extractor v2")
    print("=======================================")
    print("\n")

    print(f"📄 PDF File: {pdf_file}")
    if output_file:
        print(f"💾 Output CSV: {output_file}")

    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        return

    print("===============================\n")

    # Check for API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("⚠️ Warning: ALPHA_VANTAGE_API_KEY environment variable not set")
        print("\nSymbol classification will use fallback methods only")
    else:
        print("🔑 Using Alpha Vantage API for symbol classification")

    print("\nUsing local Ollama models for additional classification support")

    extractor = AdvancedTradeExtractor(api_key)
    extractor.process_pdf(pdf_file, output_file)


if __name__ == "__main__":
    main()
