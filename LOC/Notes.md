LOC/ contains all source code for: i) extracting newspaper pages & urls as text containing, "negro"; extracting quotations with a clear attributable speaker==negro; and for getting chained probabilities via llama3 and GPT2 via Snellius;

newspaper pages &urls and quotations are extracted via notebook, news_quotatations_extraction.ipynb

tuples_news.json are the url, (q,m,s) for each quote, after manual correction from examining images at urls. Note: d--d instances are converted to "damned", all quotation marks are converted to " or '.

Snellius/ contains the source code for getting the chained probabilities via llama3 and GPT2 via snellius.

chains_GPT2_news.json, chains_llama3.1_news.json are the resultant probabilities for each quotation in tuples_news.json. with corresponding indices.

