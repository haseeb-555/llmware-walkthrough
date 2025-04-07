
"""     Fast Start Example #3 - Prompts & Model Catalog - how to build prompts and run inferences

    In this example, we will illustrate:

    1.  Discovery - how to discover models in the llmware ModelCatalog
    2.  Load Model - how to load a selected model from the catalog
    3.  Prompt - how to create a basic prompt and run an inference with the model

    In llmware, we use the same basic formalism for the importing of all models so once you learn the recipe
    below, you should be able to import and start inferences on just about any model in the llmware catalog.

        To run GGUF models (generally marked in the catalog as 'GGUFGenerativeModel` model category and
    with names that include 'gguf' or 'tool'), then there are no additional dependencies required to start
    running local inferences.

    However, pytorch-based models will require additional dependencies to be installed, specifically:

        `pip3 install torch`
        `pip3 install transformers`

    To use an OpenAI model, you will need to `pip3 install openai`.

"""


import time
from llmware.prompts import Prompt
from llmware.models import ModelCatalog


def hello_world_questions():

    """ This is a set of useful test questions to do a 'hello world' but there is nothing special about the
    questions - please feel free to edit and ask your own queries with your own context passages.

    --if you are using one of the llmware models, please take note that the models have been trained to answer
    based on the information provided, so if you ask a question without passing any context passage, then
    don't be surprised if the model responds with 'Not Found.' """

    test_list = [

    {"query": "What is the total amount of the invoice?",
     "answer": "$22,500.00",
     "context": "Services Vendor Inc. \n100 Elm Street Pleasantville, NY \nTO Alpha Inc. 5900 1st Street "
                "Los Angeles, CA \nDescription Front End Engineering Service $5000.00 \n Back End Engineering"
                " Service $7500.00 \n Quality Assurance Manager $10,000.00 \n Total Amount $22,500.00 \n"
                "Make all checks payable to Services Vendor Inc. Payment is due within 30 days."
                "If you have any questions concerning this invoice, contact Bia Hermes. "
                "THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"},

    {"query": "What was the amount of the trade surplus?",
     "answer": "62.4 billion yen ($416.6 million)",
     "context": "Japan’s September trade balance swings into surplus, surprising expectations"
                "Japan recorded a trade surplus of 62.4 billion yen ($416.6 million) for September, "
                "beating expectations from economists polled by Reuters for a trade deficit of 42.5 "
                "billion yen. Data from Japan’s customs agency revealed that exports in September "
                "increased 4.3% year on year, while imports slid 16.3% compared to the same period "
                "last year. According to FactSet, exports to Asia fell for the ninth straight month, "
                "which reflected ongoing China weakness. Exports were supported by shipments to "
                "Western markets, FactSet added. — Lim Hui Jie"},

    {"query": "What was Microsoft's revenue in the 3rd quarter?",
     "answer": "$52.9 billion",
     "context": "Microsoft Cloud Strength Drives Third Quarter Results \nREDMOND, Wash. — April 25, 2023 — "
                "Microsoft Corp. today announced the following results for the quarter ended March 31, 2023,"
                " as compared to the corresponding period of last fiscal year:\n· Revenue was $52.9 billion"
                " and increased 7% (up 10% in constant currency)\n· Operating income was $22.4 billion "
                "and increased 10% (up 15% in constant currency)\n· Net income was $18.3 billion and "
                "increased 9% (up 14% in constant currency)\n· Diluted earnings per share was $2.45 "
                "and increased 10% (up 14% in constant currency).\n"},

    {"query": "When did the LISP machine market collapse?",
     "answer": "1987.",
     "context": "The attendees became the leaders of AI research in the 1960s."
                "  They and their students produced programs that the press described as 'astonishing': "
                "computers were learning checkers strategies, solving word problems in algebra, "
                "proving logical theorems and speaking English.  By the middle of the 1960s, research in "
                "the U.S. was heavily funded by the Department of Defense and laboratories had been "
                "established around the world. Herbert Simon predicted, 'machines will be capable, "
                "within twenty years, of doing any work a man can do'.  Marvin Minsky agreed, writing, "
                "'within a generation ... the problem of creating 'artificial intelligence' will "
                "substantially be solved'. They had, however, underestimated the difficulty of the problem.  "
                "Both the U.S. and British governments cut off exploratory research in response "
                "to the criticism of Sir James Lighthill and ongoing pressure from the US Congress "
                "to fund more productive projects. Minsky's and Papert's book Perceptrons was understood "
                "as proving that artificial neural networks approach would never be useful for solving "
                "real-world tasks, thus discrediting the approach altogether.  The 'AI winter', a period "
                "when obtaining funding for AI projects was difficult, followed.  In the early 1980s, "
                "AI research was revived by the commercial success of expert systems, a form of AI "
                "program that simulated the knowledge and analytical skills of human experts. By 1985, "
                "the market for AI had reached over a billion dollars. At the same time, Japan's fifth "
                "generation computer project inspired the U.S. and British governments to restore funding "
                "for academic research. However, beginning with the collapse of the Lisp Machine market "
                "in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began."},

    {"query": "When will employment start?",
     "answer": "April 16, 2012.",
     "context": "THIS EXECUTIVE EMPLOYMENT AGREEMENT (this “Agreement”) is entered "
                "into this 2nd day of April, 2012, by and between Aphrodite Apollo "
                "(“Executive”) and TestCo Software, Inc. (the “Company” or “Employer”), "
                "and shall become effective upon Executive’s commencement of employment "
                "(the “Effective Date”) which is expected to commence on April 16, 2012. "
                "The Company and Executive agree that unless Executive has commenced "
                "employment with the Company as of April 16, 2012 (or such later date as "
                "agreed by each of the Company and Executive) this Agreement shall be "
                "null and void and of no further effect."},

    {"query": "What is the current rate on 10-year treasuries?",
     "answer": "4.58%",
     "context": "Stocks rallied Friday even after the release of stronger-than-expected U.S. jobs data "
                "and a major increase in Treasury yields.  The Dow Jones Industrial Average gained 195.12 points, "
                "or 0.76%, to close at 31,419.58. The S&P 500 added 1.59% at 4,008.50. The tech-heavy "
                "Nasdaq Composite rose 1.35%, closing at 12,299.68. The U.S. economy added 438,000 jobs in "
                "August, the Labor Department said. Economists polled by Dow Jones expected 273,000 "
                "jobs. However, wages rose less than expected last month.  Stocks posted a stunning "
                "turnaround on Friday, after initially falling on the stronger-than-expected jobs report. "
                "At its session low, the Dow had fallen as much as 198 points; it surged by more than "
                "500 points at the height of the rally. The Nasdaq and the S&P 500 slid by 0.8% during "
                "their lowest points in the day.  Traders were unclear of the reason for the intraday "
                "reversal. Some noted it could be the softer wage number in the jobs report that made "
                "investors rethink their earlier bearish stance. Others noted the pullback in yields from "
                "the day’s highs. Part of the rally may just be to do a market that had gotten extremely "
                "oversold with the S&P 500 at one point this week down more than 9% from its high earlier "
                "this year.  Yields initially surged after the report, with the 10-year Treasury rate trading "
                "near its highest level in 14 years. The benchmark rate later eased from those levels, but "
                "was still up around 6 basis points at 4.58%.  'We’re seeing a little bit of a give back "
                "in yields from where we were around 4.8%. [With] them pulling back a bit, I think that’s "
                "helping the stock market,' said Margaret Jones, chief investment officer at Vibrant Industries "
                "Capital Advisors. 'We’ve had a lot of weakness in the market in recent weeks, and potentially "
                "some oversold conditions.'"},

    {"query": "What is the governing law?",
     "answer": "State of Massachusetts",
     "context": "19.	Governing Law and Procedures. This Agreement shall be governed by and interpreted "
                 "under the laws of the State of Massachusetts, except with respect to Section 18(a) of this Agreement,"
                 " which shall be governed by the laws of the State of Delaware, without giving effect to any "
                 "conflict of laws provisions. Employer and Executive each irrevocably and unconditionally "
                 "(a) agrees that any action commenced by Employer for preliminary and permanent injunctive relief "
                 "or other equitable relief related to this Agreement or any action commenced by Executive pursuant "
                 "to any provision hereof, may be brought in the United States District Court for the federal "
                 "district in which Executive’s principal place of employment is located, or if such court does "
                 "not have jurisdiction or will not accept jurisdiction, in any court of general jurisdiction "
                 "in the state and county in which Executive’s principal place of employment is located, "
                 "(b) consents to the non-exclusive jurisdiction of any such court in any such suit, action o"
                 "r proceeding, and (c) waives any objection which Employer or Executive may have to the "
                 "laying of venue of any such suit, action or proceeding in any such court. Employer and "
                 "Executive each also irrevocably and unconditionally consents to the service of any process, "
                 "pleadings, notices or other papers in a manner permitted by the notice provisions of Section 8."},

    {"query": "What is the amount of the base salary?",
     "answer": "$200,000.",
     "context": "2.2. Base Salary. For all the services rendered by Executive hereunder, during the "
                 "Employment Period, Employer shall pay Executive a base salary at the annual rate of "
                 "$200,000, payable semimonthly in accordance with Employer’s normal payroll practices. "
                 "Executive’s base salary shall be reviewed annually by the Board (or the compensation committee "
                 "of the Board), pursuant to Employer’s normal compensation and performance review policies "
                 "for senior level executives, and may be increased but not decreased. The amount of any "
                 "increase for each year shall be determined accordingly. For purposes of this Agreement, "
                 "the term “Base Salary” shall mean the amount of Executive’s base salary established "
                 "from time to time pursuant to this Section 2.2. "},

    {"query": "Is the expected gross margin greater than 70%?",
     "answer": "Yes, between 71.5% and 72.%",
     "context": "Outlook NVIDIA’s outlook for the third quarter of fiscal 2024 is as follows:"
                "Revenue is expected to be $16.00 billion, plus or minus 2%. GAAP and non-GAAP "
                "gross margins are expected to be 71.5% and 72.5%, respectively, plus or minus "
                "50 basis points.  GAAP and non-GAAP operating expenses are expected to be "
                "approximately $2.95 billion and $2.00 billion, respectively.  GAAP and non-GAAP "
                "other income and expense are expected to be an income of approximately $100 "
                "million, excluding gains and losses from non-affiliated investments. GAAP and "
                "non-GAAP tax rates are expected to be 14.5%, plus or minus 1%, excluding any discrete items."
                "Highlights NVIDIA achieved progress since its previous earnings announcement "
                "in these areas:  Data Center Second-quarter revenue was a record $10.32 billion, "
                "up 141% from the previous quarter and up 171% from a year ago. Announced that the "
                "NVIDIA® GH200 Grace™ Hopper™ Superchip for complex AI and HPC workloads is shipping "
                "this quarter, with a second-generation version with HBM3e memory expected to ship "
                "in Q2 of calendar 2024. "},

    {"query": "What is Bank of America's rating on Target?",
     "answer": "Buy",
     "context": "Here are some of the tickers on my radar for Thursday, Oct. 12, taken directly from "
                "my reporter’s notebook: It’s the one-year anniversary of the S&P 500′s bear market bottom "
                "of 3,577. Since then, as of Wednesday’s close of 4,376, the broad market index "
                "soared more than 22%.  Hotter than expected September consumer price index, consumer "
                "inflation. The Social Security Administration issues announced a 3.2% cost-of-living "
                "adjustment for 2024.  Chipotle Mexican Grill (CMG) plans price increases. Pricing power. "
                "Cites consumer price index showing sticky retail inflation for the fourth time "
                "in two years. Bank of America upgrades Target (TGT) to buy from neutral. Cites "
                "risk/reward from depressed levels. Traffic could improve. Gross margin upside. "
                "Merchandising better. Freight and transportation better. Target to report quarter "
                "next month. In retail, the CNBC Investing Club portfolio owns TJX Companies (TJX), "
                "the off-price juggernaut behind T.J. Maxx, Marshalls and HomeGoods. Goldman Sachs "
                "tactical buy trades on Club names Wells Fargo (WFC), which reports quarter Friday, "
                "Humana (HUM) and Nvidia (NVDA). BofA initiates Snowflake (SNOW) with a buy rating."
                "If you like this story, sign up for Jim Cramer’s Top 10 Morning Thoughts on the "
                "Market email newsletter for free. Barclays cuts price targets on consumer products: "
                "UTZ Brands (UTZ) to $16 per share from $17. Kraft Heinz (KHC) to $36 per share from "
                "$38. Cyclical drag. J.M. Smucker (SJM) to $129 from $160. Secular headwinds. "
                "Coca-Cola (KO) to $59 from $70. Barclays cut PTs on housing-related stocks: Toll Brothers"
                "(TOL) to $74 per share from $82. Keeps underweight. Lowers Trex (TREX) and Azek"
                "(AZEK), too. Goldman Sachs (GS) announces sale of fintech platform and warns on "
                "third quarter of 19-cent per share drag on earnings. The buyer: investors led by "
                "private equity firm Sixth Street. Exiting a mistake. Rise in consumer engagement for "
                "Spotify (SPOT), says Morgan Stanley. The analysts hike price target to $190 per share "
                "from $185. Keeps overweight (buy) rating. JPMorgan loves elf Beauty (ELF). Keeps "
                "overweight (buy) rating but lowers price target to $139 per share from $150. "
                "Sees “still challenging” environment into third-quarter print. The Club owns shares "
                "in high-end beauty company Estee Lauder (EL). Barclays upgrades First Solar (FSLR) "
                "to overweight from equal weight (buy from hold) but lowers price target to $224 per "
                "share from $230. Risk reward upgrade. Best visibility of utility scale names."},

    {"query": "Who is NVIDIA's partner for the driver assistance system?",
     "answer": "MediaTek",
     "context":   "Automotive Second-quarter revenue was $253 million, down 15% from the previous "
                  "quarter and up 15% from a year ago. Announced that NVIDIA DRIVE Orin™ is powering "
                  "the new XPENG G6 Coupe SUV’s intelligent advanced driver assistance system. "
                  "Partnered with MediaTek, which will develop mainstream automotive systems on "
                  "chips for global OEMs, which integrate new NVIDIA GPU chiplet IP for AI and graphics."},
  
    {
        "query": "How many medals did the U.S. win at the 2024 Summer Olympics?",
        "answer": "113 medals",
        "context": "The United States finished the 2024 Summer Olympics in Paris at the top of the medal table, winning a total of 113 medals: 39 gold, 41 silver, and 33 bronze. The event featured over 10,000 athletes from more than 200 nations competing in 32 sports. Team USA’s dominance was seen across multiple disciplines. In swimming, Katie Ledecky added 3 more golds to her tally, becoming the most decorated female swimmer in Olympic history. Simone Biles returned to gymnastics and led the U.S. team to a gold medal in the team event, while also earning individual gold in the balance beam. In track and field, Noah Lyles won gold in the 100m and 200m sprints, joining a legendary group of sprinters. The U.S. women’s soccer team also took bronze after a tough semifinal loss to Germany. With strong showings across team and individual sports, the U.S. reaffirmed its Olympic legacy in Paris."
    },

    {
        "query": "Who won the 2023 Formula 1 World Championship?",
        "answer": "Max Verstappen",
        "context": "Max Verstappen of Red Bull Racing secured the 2023 Formula 1 World Championship with six races to spare, finishing the season with a record-breaking 19 wins out of 23 races. The Dutch driver dominated the grid from the opening Grand Prix in Bahrain to the final race in Abu Dhabi. His consistent performance, combined with Red Bull’s superior car design and strategy, made him nearly untouchable. Verstappen’s teammate, Sergio Perez, finished second in the standings, securing Red Bull’s first one-two finish in the driver’s championship. The team also comfortably won the Constructors’ Championship. Despite challenges from Mercedes and Ferrari early in the season, Verstappen pulled ahead by mid-year, often winning races by margins exceeding 10 seconds. His crowning moment came at the Japanese Grand Prix in Suzuka, where he clinched the title after winning from pole. At 26 years old, Verstappen now has three consecutive world titles, drawing comparisons with legends like Schumacher and Hamilton."
    },

    {
        "query": "Who scored the winning goal in the 2023 UEFA Champions League Final?",
        "answer": "Erling Haaland",
        "context": "The 2023 UEFA Champions League Final, held at Istanbul’s Atatürk Olympic Stadium, featured a tense clash between Manchester City and Real Madrid. After a goalless first half marked by strong defensive play and brilliant saves from both Thibaut Courtois and Ederson, the deadlock was broken in the 78th minute when Erling Haaland scored the only goal of the match. The goal came from a Kevin De Bruyne through ball, with Haaland slotting it calmly past Courtois. This victory marked Manchester City’s first-ever Champions League title, completing a historic treble for the club in the 2022–23 season after already winning the Premier League and FA Cup. Haaland, who finished the Champions League campaign with 13 goals, was named Player of the Tournament. Pep Guardiola’s tactical acumen was widely praised, with many calling this one of the greatest seasons by an English club in modern football history."
    },

    {
        "query": "What was the final score in Game 7 of the 2023 World Series?",
        "answer": "Texas Rangers 5, Atlanta Braves 3",
        "context": "In one of the most dramatic Game 7s in World Series history, the Texas Rangers defeated the Atlanta Braves 5-3 to claim their first championship title. The deciding game, held at Globe Life Field in Arlington, was a nail-biter from the start. The Braves took an early lead in the second inning with a two-run homer from Matt Olson. However, the Rangers fought back with a clutch three-run home run by Corey Seager in the fourth. Pitching proved pivotal as Nathan Eovaldi delivered six strong innings for Texas, allowing only three earned runs and striking out eight. The bullpen held firm, with closer José Leclerc sealing the win in the ninth. MVP honors went to Seager, who hit .375 across the series with 3 home runs and 9 RBIs. The win capped off a fairytale postseason for the Rangers, who entered the playoffs as a Wild Card team and beat the Orioles, Astros, and finally the Braves in a stunning run."
    },

    {
        "query": "Who was the MVP of the 2024 NCAA Men's Basketball Tournament?",
        "answer": "Zach Edey",
        "context": "The 2024 NCAA Men’s Basketball Tournament concluded with Purdue University winning its first national championship, defeating the University of Connecticut 72-68 in a tightly contested final. The standout player of the tournament was Purdue’s 7'4\" center Zach Edey, who was named the tournament’s Most Outstanding Player. Edey averaged 27.3 points, 12.5 rebounds, and 3.2 blocks per game throughout March Madness. In the championship game, he scored 29 points and grabbed 14 rebounds, dominating the paint and drawing multiple fouls that kept UConn’s big men in foul trouble. Purdue's road to the championship included wins over Tennessee, Marquette, and a dramatic overtime victory against Kansas in the Final Four. Head Coach Matt Painter praised the team’s resilience and Edey’s leadership, calling him “the heart of this team.” Edey’s performance sparked discussions about his future in the NBA, with many analysts projecting him as a lottery pick in the upcoming draft."
    }



    ]

    return test_list


def fast_start_prompting  (model_name):

    """ This is the main example script - it loads the question list, loads the model and executes the prompts. """

    t0 = time.time()

    # load in the 'hello world' test questions above
    test_list = hello_world_questions()

    print(f"\n > Loading Model: {model_name}...")

    prompter = Prompt().load_model(model_name)

    t1 = time.time()
    print(f"\n > Model {model_name} load time: {t1-t0} seconds")
 
    for i, entries in enumerate(test_list):
        print(f"\n{i+1}. Query: {entries['query']}")
     
        #   run the prompt
        output = prompter.prompt_main(entries["query"],
                                      context=entries["context"],
                                      prompt_name="default_with_context",
                                      temperature=0.30)

        #   'output' is a dictionary with two keys - 'llm_response' and 'usage'
        #   --'llm_response' is the output from the model
        #   --'usage' is a dictionary with the usage stats

        llm_response = output["llm_response"].strip("\n")
        print(f"LLM Response: {llm_response}")

        #   note: the 'gold answer' is the answer we provided above in the hello_world question list
        print(f"Gold Answer: {entries['answer']}")

        print(f"LLM Usage: {output['usage']}")

    t2 = time.time()
    print(f"\nTotal processing time: {t2-t1} seconds")

    return 0


if __name__ == "__main__":

    #   Step 1 - we will pick a model from the ModelCatalog

    #   A few useful methods to discover and display a list of available models...

    #   all generative models
    llm_models = ModelCatalog().list_generative_models()

    #   if you only want to see the local models
    llm_local_models = ModelCatalog().list_generative_local_models()

    #   to see only the open source models
    llm_open_source_models = ModelCatalog().list_open_source_models()

    #   we will print out the local models
    for i, models in enumerate(llm_local_models):
        print("models: ", i, models["model_name"], models["model_family"])

    #   for purposes of demo, try a few selected models from the list

    #   each of these pytorch models are ~1b parameters and will run reasonably fast and accurate on CPU
    #   --per note above, may require separate pip3 install of: torch and transformers
    pytorch_generative_models = ["llmware/bling-1b-0.1", "llmware/bling-tiny-llama-v0", "llmware/bling-falcon-1b-0.1"]

    #   bling-answer-tool is 1b parameters quantized
    #   bling-phi-3-gguf is 3.8b parameters quantized
    #   dragon-yi-6b-gguf is 6b parameters quantized
    gguf_generative_models = ["bling-answer-tool", "bling-phi-3-gguf","llmware/dragon-yi-6b-gguf"]

    #   by default, we will select a gguf model requiring no additional imports
    model_name = gguf_generative_models[2]

    #   to swap in a gpt-4 openai model - uncomment these two lines
    #   model_name = "gpt-4"
    #   os.environ["USER_MANAGED_OPENAI_API_KEY"] = "<insert-your-openai-key>"

    fast_start_prompting(model_name)
