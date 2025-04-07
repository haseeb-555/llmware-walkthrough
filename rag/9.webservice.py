"""
Fast Start #9 - Function Calls with Web Services

Combines function calls with external web services to drive complex automation.

Models:
1. slim-extract-tool
2. slim-summary-tool
3. bling-stablelm-3b-tool

Web Services:
1. Yfinance - stock ticker
2. Wikipedia - company background information

NOTE: To run this example, please install yfinance: pip install yfinance
"""

from llmware.web_services import YFinance
from llmware.models import ModelCatalog
from llmware.parsers import WikiParser
from importlib import util
import json

if not util.find_spec("yfinance"):
    print("\nTo run this example, please install yfinance: pip install yfinance")

text = (
    "BEAVERTON, Ore.--(BUSINESS WIRE)--NIKE, Inc. (NYSE:NKE) today reported fiscal 2024 financial results for its "
    "third quarter ended February 29, 2024.) “We are making the necessary adjustments to drive NIKE’s next chapter "
    "of growth Post this Third quarter revenues were slightly up on both a reported and currency-neutral basis* "
    "at $12.4 billion NIKE Direct revenues were $5.4 billion, slightly up on a reported and currency-neutral basis "
    "NIKE Brand Digital sales decreased 3 percent on a reported basis and 4 percent on a currency-neutral basis "
    "Wholesale revenues were $6.6 billion, up 3 percent on a reported and currency-neutral basis Gross margin "
    "increased 150 basis points to 44.8 percent, including a detriment of 50 basis points due to restructuring charges "
    "Selling and administrative expense increased 7 percent to $4.2 billion, including $340 million of restructuring "
    "charges Diluted earnings per share was $0.77, including $0.21 of restructuring charges. Excluding these "
    "charges, Diluted earnings per share would have been $0.98* “We are making the necessary adjustments to "
    "drive NIKE’s next chapter of growth,” said John Donahoe, President & CEO, NIKE, Inc. “We’re encouraged by "
    "the progress we’ve seen, as we build a multiyear cycle of new innovation, sharpen our brand storytelling and "
    "work with our wholesale partners to elevate and grow the marketplace."
)

def research_example1():
    model = ModelCatalog().load_model("slim-extract-tool", temperature=0.0, sample=False)
    model2 = ModelCatalog().load_model("slim-summary-tool", sample=False, temperature=0.0, max_output=200)
    model3 = ModelCatalog().load_model("bling-stablelm-3b-tool", sample=False, temperature=0.0)

    research_summary = {}

    extract_keys = [
        "stock ticker", "company name", "total revenues",
        "restructuring charges", "digital growth",
        "ceo comment", "quarter end date"
    ]

    print("\nStep 1 - Extract information from source text")
    for key in extract_keys:
        response = model.function_call(text, params=[key])
        dict_key = key.replace(" ", "_")
        print(f"Extracting - {key} - {response['llm_response']}")
        values = response["llm_response"].get(dict_key)
        if isinstance(values, list) and values:
            research_summary[dict_key] = values[0]
        else:
            print(f"Warning: '{dict_key}' not found or empty.")

    print("\nStep 2 - Web lookup via YFinance")
    if "stock_ticker" in research_summary:
        ticker = research_summary["stock_ticker"].split(":")[-1]
        try:
            yf = YFinance().get_stock_summary(ticker=ticker)
            research_summary.update({
                "current_stock_price": yf.get("currentPrice", "N/A"),
                "high_ltm": yf.get("fiftyTwoWeekHigh", "N/A"),
                "low_ltm": yf.get("fiftyTwoWeekLow", "N/A"),
                "trailing_pe": yf.get("trailingPE", "N/A"),
                "forward_pe": yf.get("forwardPE", "N/A"),
                "volume": yf.get("volume", "N/A")
            })

            yf2 = YFinance().get_financial_summary(ticker=ticker)
            research_summary.update({
                "market_cap": yf2.get("marketCap", "N/A"),
                "price_to_sales": yf2.get("priceToSalesTrailing12Months", "N/A"),
                "revenue_growth": yf2.get("revenueGrowth", "N/A"),
                "ebitda": yf2.get("ebitda", "N/A"),
                "gross_margin": yf2.get("grossMargins", "N/A"),
                "currency": yf2.get("currency", "N/A")
            })

            yf3 = YFinance().get_company_summary(ticker=ticker)
            research_summary.update({
                "sector": yf3.get("sector", "N/A"),
                "website": yf3.get("website", "N/A"),
                "industry": yf3.get("industry", "N/A"),
                "employees": yf3.get("fullTimeEmployees", "N/A")
            })

            execs = []
            for officer in yf3.get("companyOfficers", []):
                name = officer.get("name", "N/A")
                title = officer.get("title", "N/A")
                age = officer.get("age", "N/A")
                pay = officer.get("totalPay", "N/A")
                execs.append((name, title, age, pay))
            research_summary["officers"] = execs

        except Exception as e:
            print("YFinance lookup failed:", e)

    print("\nStep 3 - Wikipedia company lookup")
    if "company_name" in research_summary:
        company_name = research_summary["company_name"]
        wiki_data = WikiParser().add_wiki_topic(company_name, target_results=1)

        company_overview = ""
        for i, block in enumerate(wiki_data.get("blocks", [])):
            if i < 3:
                company_overview += block.get("text", "")

        if company_overview:
            print("-- Summarizing Wikipedia content")
            summary = model2.function_call(company_overview, params=["company history (5)"])
            research_summary["summary"] = summary.get("llm_response", "N/A")

            print("-- Extracting founding date")
            fd_response = model.function_call(company_overview, params=["founding date"])
            research_summary["founding_date"] = fd_response["llm_response"].get("founding_date", ["N/A"])[0]

            print("-- Extracting company description")
            cd_response = model.function_call(company_overview, params=["company description"])
            research_summary["company_description"] = cd_response["llm_response"].get("company_description", ["N/A"])[0]

            print("-- Answering questions via bling model")
            business = model3.inference("What is an overview of company's business?", add_context=company_overview)
            research_summary["business_overview"] = business.get("llm_response", "N/A")

            origin = model3.inference("What is the origin of the company's name?", add_context=company_overview)
            research_summary["origin_of_name"] = origin.get("llm_response", "N/A")

            products = model3.inference("What are the product names", add_context=company_overview)
            research_summary["products"] = products.get("llm_response", "N/A")
        else:
            print("No Wikipedia overview available.")

    print("\nStep 4 - Completed Research Summary")
    for idx, (k, v) in enumerate(research_summary.items(), 1):
        val = str(v).replace("\n", "").replace("\r", "").replace("\t", "")
        print(f"\t -- {idx} - {k.ljust(25)} - {val[:100]}")

    return research_summary


if __name__ == "__main__":
    research_example1()
