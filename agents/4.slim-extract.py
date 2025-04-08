
""" This example illustrates how to use the slim-extract model to extract custom keys from selected text.
    We have included a set of sample earnings releases (comprising lines ~10 - ~385 of this script), and run a
    simple loop through the earnings releases, showing how to create an extract prompt to identify
    'revenue growth' from these examples.

    There are several function-calling models in the slim-extract family, fine-tuned on multiple leading
    small model base foundations - full list and options are below in the code.  """

from llmware.models import ModelCatalog

# Sample earnings releases

earnings_releases = [

 {"context": "Adobe shares tumbled as much as 11% in extended trading Thursday after the design software maker "
    "issued strong fiscal first-quarter results but came up slightly short on quarterly revenue guidance. "
    "Here’s how the company did, compared with estimates from analysts polled by LSEG, formerly known as Refinitiv: "
    "Earnings per share: $4.48 adjusted vs. $4.38 expected Revenue: $5.18 billion vs. $5.14 billion expected "
    "Adobe’s revenue grew 11% year over year in the quarter, which ended March 1, according to a statement. "
    "Net income decreased to $620 million, or $1.36 per share, from $1.25 billion, or $2.71 per share, "
    "in the same quarter a year ago. During the quarter, Adobe abandoned its $20 billion acquisition of "
    "design software startup Figma after U.K. regulators found competitive concerns. The company paid "
    "Figma a $1 billion termination fee."},

 {"context": "Dick’s Sporting Goods raised its dividend by 10% on Thursday as the company posted its largest sales "
    "quarter in its history and projected another year of growth. The company’s shares jumped more than "
    "15% in intraday trading. CEO Lauren Hobart said on an earnings call Thursday that Dick’s sales "
    "growth came from bigger tickets — either higher prices or more expensive items — as its transactions "
    "were flat. Many retailers benefited from a 53rd week in fiscal 2023, but Dick’s said it still broke "
    "records during its fiscal fourth quarter even without those extra days. Here’s how the athletic "
    "apparel retailer did compared with what Wall Street was anticipating, based on a survey of "
    "analysts by LSEG, formerly known as Refinitiv: Earnings per share: $3.85 adjusted vs. $3.35 expected "
    "Revenue: $3.88 billion vs. $3.80 billion expected The company’s reported net income for the three-month "
    "period that ended Feb. 3 was $296 million, or $3.57 per share, compared with $236 million, or $2.60 a "
    "share, a year earlier. Excluding one-time items related to impairment charges and inventory write-offs, "
    "Dick’s reported earnings per share of $3.85. Sales rose to $3.88 billion, up about 8% from $3.60 billion "
    "a year earlier. “With our industry-leading assortment and strong execution, we capped off the year "
    "with an incredibly strong fourth quarter and holiday season,” Hobart said in a statement. “We are "
    "guiding to another strong year in 2024. We plan to grow both our sales and earnings through "
    "positive comps, higher merchandise margin and productivity gains,” she added. During the quarter, "
    "same-store sales rose 2.8%, well ahead of the 0.8% lift that analysts had expected, according to "
    "StreetAccount. “Growth in transactions” and market share gains drove the increase, said Executive "
    "Chairman Ed Stack."},

 {"context": "Comcast topped both revenue and profit estimates in the fourth quarter as it lost fewer broadband "
    "subscribers than expected, and it raised its dividend 7%, the company said Thursday. "
   "Here’s how Comcast performed, compared with estimates from analysts surveyed by LSEG, "
   "formerly known as Refinitiv.  Earnings per share: 84 cents adjusted vs. 79 cents expected  "
   "Revenue: $31.25 billion vs. $30.51 billion expected For the quarter ended Dec. 31, net "
   "income rose 7.8% to $3.26 billion, or 81 cents a share, compared with $3.02 billion, or "
   "70 cents a share, a year earlier. Revenue increased 2.3% compared with the prior-year period. "
   "Adjusted earnings before interest, taxes, depreciation and amortization (EBITDA) was flat year "
   "over year at about $8 billion.   'For the third consecutive year, we generated the highest revenue, "
   "adjusted EBITDA and adjusted EPS in our company’s history', Comcast Chief Executive Officer Brian "
   "Roberts said in a statement. 'We also reported the highest adjusted EBITDA on record at Theme Parks; "
   "were the #1 studio in worldwide box office for the first time since 2015; and maintained Peacock’s "
   "position as the fastest growing streamer in the U.S.'"},

 {"context": "Dollar General forecast annual sales above Wall Street estimates on Thursday, banking on higher "
     "demand from inflation-hit customers buying groceries and essentials from the discount retailer’s stores.  "
     "Shares of the company rose about 6% in early trading, after falling nearly 45% in 2023 on rising costs "
     "and stiff competition from bigger retailers. But higher prices and borrowing costs have prompted "
     "budget-conscious consumers to cook more meals at home, helping Dollar General record stronger "
     "footfall at its outlets as shoppers hunt for lower-margin, needs-based goods, over pricier general "
     "merchandise. “With customer traffic growth and market share gains during the quarter, we believe our "
     "actions are resonating with customers,” CEO Todd Vasos said in a statement. Vasos’s strategy - to focus "
     "on the basics, like more employee presence at stores, greater customer engagement and expanding "
     "private-label brands - has helped stabilize Dollar General’s business. Over the last few quarters, "
     "Dollar General and rival Dollar Tree have struggled with rising costs linked to their supply "
     "chains, labor and raw materials, while facing tough competition from retailers like Walmart "
     "and Chinese ecommerce platform Temu. Dollar Tree’s shares fell more than 15% on Wednesday, after it "
     "forecast weak sales and profit for 2024 and laid out plans to shutter 970 of its Family Dollar "
     "stores. “Dollar General has a much rosier outlook than Dollar Tree... Dollar Tree’s challenges "
     "with Family Dollar were years in the making, while Dollar General has embarked on an aggressive "
     "effort to add more frozen, refrigerated and fresh produce,” eMarketer senior analyst Zak Stambor said.  "
     "Dollar General forecast 2024 sales to grow between 6.0% and 6.7%, above analysts’ estimate of 4.4% "
     "growth to $40.33 billion, according to LSEG data. It still sees annual per-share profit between "
     "$6.80 and $7.55, compared with estimates of $7.55.  Its fourth-quarter net sales of $9.86 billion "
     "surpassed estimates of $9.78 billion. It also reported an estimate-beating profit of $1.83 per share."},


    {"context":  "Best Buy surpassed Wall Street’s revenue and earnings expectations for the holiday quarter on "
                 "Thursday, even as the company navigated through a period of tepid consumer electronics demand.  "
                 "But the retailer warned of another year of softer sales and said it would lay off workers and "
                 "cut other costs across the business. CEO Corie Barry offered few specifics, but said the "
                 "company has to make sure its workforce and stores match customers’ changing shopping habits. "
                 "Cuts will free up capital to invest back into the business and in newer areas, such as artificial "
                 "intelligence, she added. “This is giving us some of that space to be able to reinvest into "
                 "our future and make sure we feel like we are really well positioned for the industry to "
                 "start to rebound,” she said on a call with reporters. For this fiscal year, Best Buy anticipates "
                 "revenue will range from $41.3 billion to $42.6 billion. That would mark a drop from the most "
                 "recently ended fiscal year, when full-year revenue totaled $43.45 billion. It said comparable "
                 "sales will range from flat to a 3% decline. The retailer plans to close 10 to 15 stores "
                 "this year after shuttering 24 in the past fiscal year. One challenge that will affect sales "
                 "in the year ahead: it is a week shorter. Best Buy said the extra week in the past fiscal "
                 "year lifted revenue by about $735 million and boosted diluted earnings per share by about "
                 "30 cents. Shares of Best Buy closed more than 1% higher Thursday after briefly touching "
                 "a 52-week high of $86.11 earlier in the session. Here’s what the consumer electronics "
                 "retailer reported for its fiscal fourth quarter of 2024 compared with what Wall Street was "
                 "expecting, based on a survey of analysts by LSEG, formerly known as Refinitiv: "
                 "Earnings per share: $2.72, adjusted vs. $2.52 expected Revenue: $14.65 billion vs. $14.56 "
                 "billion expected A dip in demand, but a better-than-feared holiday Best Buy has dealt "
                 "with slower demand in part due to the strength of its sales during the pandemic. Like "
                 "home improvement companies, Best Buy saw outsized spending as shoppers were stuck at "
                 "home. Plus, many items that the retailer sells like laptops, refrigerators and home "
                 "theater systems tend to be pricier and less frequent purchases. The retailer has cited other "
                 "challenges, too: Shoppers have been choosier about making big purchases while dealing "
                 "with inflation-driven higher prices of food and more. Plus, they’ve returned to "
                 "splitting their dollars between services and goods after pandemic years of little "
                 "activity. Even so, Best Buy put up a holiday quarter that was better than feared. "
                 "In the three-month period that ended Feb. 3, the company’s net income fell by 7% to "
                 "$460 million, or $2.12 per share, from $495 million, or $2.23 per share in the year-ago "
                 "period. Revenue dropped from $14.74 billion a year earlier. Comparable sales, a metric that "
                 "includes sales online and at stores open at least 14 months, declined 4.8% during the "
                 "quarter as shoppers bought fewer appliances, mobile phones, tablets and home theater "
                 "setups than the year-ago period. Gaming, on the other hand, was a strong sales "
                 "category in the holiday quarter."}

]

#   *** Execution script starts here ***

#   specialized function-calling extraction models on top of several leading small model bases,
#   ranging from 0.5b (qwen2) - 3.8b (phi3)

slim_extract_models = ["slim-extract-tool",                 #   original - stablelm-3b (2.7b)
                       "slim-extract-tiny-tool",            #   tiny-llama 1.1b
                       "slim-extract-qwen-1.5b-gguf",       #   **NEW** qwen 1.5b
                       "slim-extract-phi-3-gguf",           #   **NEW** phi-3 (3.8b)
                       "slim-extract-qwen-0.5b-gguf"]       #   **NEW** qwen 0.5b

#   load the model
model = ModelCatalog().load_model("slim-extract-tool",sample=False,temperature=0.0, max_output=100)

#   iterate through the earnings release samples above
for i, sample in enumerate(earnings_releases):

    #   key line: execute function_call on selected model with 'custom_key' = "revenue growth"
    response = model.function_call(sample["context"], function="extract", params=["revenue growth"])

    #   display the response on the screen
    print("extract response: ", i, response["llm_response"])
