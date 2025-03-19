# Augmented search

This is my quick-and-dirty project for using LLMs in order to augment the search of scientific collaborators
by automatically retrieving relevant links, summarizing their research interests and score them against the topics I am interested in. 
It is primarily developed for the personal use, though I might make it more open-ended in future.


## Workflow

The script takes a csv table as an input. It must have a single column named "name", e.g.:

```csv
name
Albert Einstein
Richard Sutton
John von Neumann
Merlin Monroe
Max Planck
Yan GoodFellow
```

Another optional input is `label` -- a piece of query common for all the names (I usually use the institute name or something like "researcher"). 
For a given name a search query is composed of a name and label, than this query is sent to a search engine and a few top results are retrieved. 
The text content from these websites is composed into a single query which is send send to an LLM as well 
as the instructions summarize the research interest as 5 keywords.

Once the keywords are obtained,they are scored against the `score_query` with `SentenceTransformers` (thanks, HuggingFace) -- 
larger score meaning more semantic similarity.

The scrip can be run as follows:

```
augsearch --db out.db --num_websites 4 --label "Researcher"  --out_csv out.csv --names names.csv --llm_params gemini_params.js --score_query "computer vision, reinforcement learning, healthcare"
```

All the results are stored in SQLite database. If the process is interrupted, it can be rerun starting from the last successful name.

## TODO list

* Summarize each website separately and than aggregate the results.
* Better website contents cleaning 
* Implement chunking for large queries
* Fix strange search engines outputs
* Add more search engines
* Add more llm providers