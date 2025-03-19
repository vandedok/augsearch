import pandas as pd
from tqdm.auto import tqdm
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from googlesearch import search
from bs4 import BeautifulSoup
from ollama import Client as OllamaClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup, Comment
import unicodedata
import html
import re
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()
try:
    from google import genai
except:
    logger.info("Couldn't import Google genai")



class LLMConnector:
    def ask(self, query, context_window=None):
        raise NotImplementedError
    
class OllamaConnector(LLMConnector):
    
    def __init__(self, params):
        self.client = OllamaClient(host=params["host"], headers=params["headers"])
        self.context_window = params["context_window"]
        self.model = params["model"]    
        self.params = params
 
    
    def ask(self, query, context_window=None):
        if context_window is None:
            logger.info(f"Using defualt context window of {context_window}")
            context_window = self.context_window
        else:
            logger.info(f"Using on-the-run-defined context window of {context_window}")
        
        check_num_tokens(text=query, context_window=context_window)
        
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            options={"num_ctx": context_window},
        )

        return response.message.content
    
class GeminiLLMConnector(LLMConnector):
    def __init__(self, params):
        self.client = genai.Client(api_key=params["api_key"])
        self.model = params["model"]
        self.max_tokens = self.client.models.get(model=self.model).input_token_limit
        self.max_tokens * 0.9

    def ask(self, query, context_window=None):

        num_tokens = self.client.models.count_tokens(model=self.model,contents=query).total_tokens
        if num_tokens > self.max_tokens:
            logger.warning(f"Number of tokens in query ({num_tokens}) is bigger than the allowed maxiumum of {self.max_tokens}. Clipping the query")
            query = query[:self.max_tokens]
        response = self.client.models.generate_content(
            model=self.model,
            contents=query
        )

        return response.text


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)

def get_llmbot_connector(params):
    if params["which"] == "ollama":
        # f"http://localhost:{port}"
        # return Client(host=params["host"], headers=params["headers"])
        return OllamaConnector(params)
    elif params["which"] == "gemini":
        return GeminiLLMConnector(params)
    else:
        raise ValueError(f"Unknown credentials llm: credentials['which']={params["which"]}")


def clean_html_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unnecessary tags
    for unwanted in soup(
        [
            "script",
            "style",
            "header",
            "footer",
            "nav",
            "aside",
            "noscript",
            "iframe",
            "embed",
            "object",
            "form",
            "input",
            "button",
            "meta",
            "link",
            "svg",
            "canvas",
        ]
    ):
        unwanted.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove elements with display: none, hidden, or opacity: 0
    for tag in soup.find_all(attrs={"style": lambda v: v and ("display: none" in v.lower() or "visibility: hidden" in v.lower() or "opacity: 0" in v.lower())}):
        tag.decompose()

    # Remove text like cookie notices
    for tag in soup.find_all(string=lambda text: text and "cookies" in text.lower()):
        tag.extract()

    # Extract and clean text, separate elements by newlines
    text = soup.get_text(separator="\n")

    # Normalize whitespace and remove empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Handle smart quotes and special Unicode
    clean_text = "\n".join(lines)
    clean_text = unicodedata.normalize("NFKD", clean_text)

    # Unescape HTML entities
    clean_text = html.unescape(clean_text)

    # Remove large number blocks (optional)
    clean_text = re.sub(r"\b\d{5,}\b", "", clean_text)

    return clean_text


def check_num_tokens(
    text,
    context_window=8000,
    tokenizer=AutoTokenizer.from_pretrained("google-bert/bert-base-cased", verbose=False),
):
    tokens = tokenizer.tokenize(text, verbose=False)
    num_tokens = len(tokens)

    if hasattr(tokenizer, "name_or_path"):
        tokenizer_name = tokenizer.name_or_path
    else:
        tokenizer_name = "Unknown"
    if num_tokens > context_window:
        logger.warning(f"The text might be too long. Num tokens:{num_tokens}. Context window: {context_window}. Tokenizer: {tokenizer_name}")
        return False, num_tokens
    else:
        logger.info(f"The text length most likely is fine. Num tokens:{num_tokens}. Context window {context_window}. Tokenizer: {tokenizer_name}")
        return True, num_tokens


def get_site_contents(url, lines_set=True, use_headers=True, use_selenium=False):

    if url.endswith(".pdf"):
        logger.error(f"PDFs are currently not supported")
        return [], []

    try:
        if use_selenium:
            options = Options()
            options.add_argument("--headless")  # Run without GUI
            options.add_argument("--disable-blink-features=AutomationControlled")  
            options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
            service = Service("/usr/bin/chromedriver") 
            driver = webdriver.Chrome(service=service, options=options)
            driver.get("https://pure.itu.dk/en/persons/elisa-mekler")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()
        else:
            session = requests.Session()
            if use_headers:
                    headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "same-origin",
                }
            session.headers.update(headers)
            response = session.get(url)
            # response = requests.get(url, headers=headers if use_headers else None)
            if response.status_code != 200:
                return [f"Got response with status code: {response.status_code}"], []
            soup = BeautifulSoup(response.text,  "html.parser")
        if lines_set:
            lines = set()
        else:
            lines = list()
        for el in soup.find_all():
            text = el.get_text(strip=True)
            if text:
                if lines_set:
                    lines.add(text)
                else:
                    lines.append(text)
        lines = list(lines)
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        return lines, links
    except Exception as e:
        logger.error(f"Failed to reach url: {url} due to: {e}. Returning empty results")
        return [], []
    


def get_links(name, label="", num_websites=3):
    search_query = f"{name} + {label}"
    results = list(search(search_query, stop=num_websites))

    links_content_list = []
    for url in results:
        lines_url, links_url = get_site_contents(url)
        links_content_list.append((url, "\n".join(lines_url)))
        # lines_dict[url] = '\n'.join(lines_url)
    return links_content_list


def clean_with_llmbot(text, llm_connector):
    query = f"""
    This is the text from a website related to a researcher.  
    Try figure out their reserach interests. Filter out the garbage, but keep all the relevant to theire research topics, interests, publications and so on.
    The text:
    <<<
    {text}
    >>>
    """


    response = llm_connector.ask(query=query)
    return response


def summarize_with_llmbot(links_contents, name, llm_connector):
    num_websites = len(links_contents)
    links = [x[0] for x in links_contents]
    query = f"""
    These are several websites summaries ({num_websites} and the corresponding links. 
    All the websites are presumably related to a researcher, their name is {name}.
    Please, figure out what their scientific interests are. Provide the answer as 5 keywords and a short (3-5 sentences) summary.
    The final answer must consist of 6 lines in total: 5 lines with the keywords, one line for the summary.
    Don't put any numbers or titles or anything else. No empty lines in between. The final format is really important.

    The template:
    keyword_1
    keyword_2
    keyword_3
    keyword_4
    keyword_5
    short_summary

    The links and texts are the following:

    """

    for link_n, (link, text) in enumerate(links_contents):
        link_query = f"\nLink #{link_n}: {link}:\n\n<<<\n{text}\n>>>\n\n"
        query += link_query

    query += f"""
    Please, figure out what their scientific interests are. Provide the answer as 5 keywords and a short (3-5 sentences) summary.
    The final answer must consist of 6 lines in total: 5 lines with the keywords, one line for the summary.
    Don't put any numbers or titles or anything else. No empty lines in between. The final format is really important.

    The template:
    keyword_1
    keyword_2
    keyword_3
    keyword_4
    keyword_5
    short_summary
    """
    response = llm_connector.ask(query=query)
    return response


def fix_format_with_llmbot(inputs, name, llm_connector):
    query = f"""
    This text should contain 5 keywords and a link. Put them in the following format:
    1. There must be exactly 5 lines
    2. Each keyword must be on its own line
    4 The link must be on its own line
    Don't add any titiles, lines numbers or any other formatting beyond specified.
    If you can't get the kewords or the link, put the NaN on this line, but keep the format

    This is the text:
    <<<
    inputs
    >>>
    """
    response = llm_connector(query=query)
    return response


def parse_summary(summary):
    split = summary.strip().split("</think>")[-1].split("\n")
    split = [x for x in split if len(x) if x!=""]
    try:
        assert len(split) == 6
        kws = ", ".join(split[:5])
        short_summary = split[5]
        return kws, short_summary
    except:
        logger.warning("Failed to parse keywords in line by line mode")
    try:
        assert len(split) == 2
        kws = split[0]
        short_summary = split[1]
        return kws, short_summary
    except:
        logger.warning("Failed to parse keywords in 2 line mode")
    try:
        assert len(split)==1
        split = split[0].split(", ")
        kws = ", ".join(split[:5])
        summary = ", ".join(split[5:])
    except:
        logger.warning("Failed to parse keywors in comma separated mode")

    kws = ""
    short_summary = summary.replace("\n", ", ")
    logger.warning(f"Failed to parse the formatting Len(spilt)={len(split)}. Split:\n {split}\n ")
    return kws, short_summary


def prepare_table(inputs_df, con, table_name, num_websites, if_exists="skip"):
    tables_df = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table';", con)
    exists = table_name in list(tables_df.name)

    if exists and if_exists == "skip":
        print("Table exists, doing nothing")
    else:
        if exists and "if_exists" == "replace":
            logger.warning(f"Repacing already present table: {table_name}")
        fields = ["name", "score", "keywords", "short_summary"] + [f"link_{x}" for x in range(num_websites)]
        fields_empty_on_init = [x for x in fields if x not in ["name", "score"]]
        df = pd.DataFrame(columns=fields)

        df["name"] = inputs_df.name
        df = df.assign(**{x: "" for x in fields_empty_on_init})
        df.to_sql(name=table_name, con=con, if_exists="fail" if if_exists == "skip" else if_exists)


def get_links(con, table_name, label, num_websites):
    # names_no_links, _ = get_names_to_process(df_names, con=con, table_name=table_name, column="link_0")
    names_no_links = pd.read_sql(f"SELECT name FROM {table_name} WHERE link_0 == '';", con=con).name
    cur = con.cursor()
    for name in tqdm(names_no_links, desc="Getting the links"):
        search_query = f"{name} + {label}"
        results = list(search(search_query, num_results=num_websites))
        assert True
        clause_line = ", ".join([f"link_{i} = ?" for i in range(len(results))])
        cur.execute(
            f"UPDATE {table_name} SET {clause_line} WHERE name = ?",
            tuple(results) + (name,),
        )
        con.commit()
    return names_no_links


def get_keywords_and_summaries(con, table_name, num_websites, llm_connector, score_query):
    fields = ["name"] + [f"link_{x}" for x in range(num_websites)]
    fields = ", ".join(fields)
    names_links_no_kws = pd.read_sql(
        f"SELECT {fields} FROM {table_name} WHERE keywords == '' AND link_0 != ''",
        con=con,
    )
    cursor = con.cursor()
    summaries = []
    st_model=SentenceTransformer("all-mpnet-base-v2")

    pbar = tqdm(
        names_links_no_kws.iterrows(),
        total=len(names_links_no_kws),
        desc="Getting keywords, summaries and scores:",
    )
    for idr, row in pbar:
        pbar.set_description(row["name"][:15])
        # print("row.name: ", row["name"])
        links_contents = []
        for website_i in range(num_websites):
            url = row[f"link_{website_i}"]
            contents, _ = get_site_contents(url)
            contents_cleaned = clean_html_text("\n".join(contents))
            ollama_cleaned_content = clean_with_llmbot(
                contents_cleaned,
                llm_connector=llm_connector,
            )
            links_contents.append((url, ollama_cleaned_content))
        name = row["name"]
        summary = summarize_with_llmbot(
            links_contents,
            name=name,
            llm_connector=llm_connector,
        )
        summaries.append(summary)
        kws, short_summary = parse_summary(summary)
        # print(f"UPDATE {table_name} SET keywords = '{kws}', best_link = '{best_link}' WHERE name = '{name}';")
        score = get_single_score(key=kws, query=score_query, st_model=st_model)
        cursor.execute(
            f"UPDATE {table_name} SET score = ?, keywords = ?, short_summary = ? WHERE name = ?;",
            (score, kws, short_summary, name),
        )
        con.commit()

def get_single_score(
        key,
        query,
        st_model=SentenceTransformer("all-mpnet-base-v2"),
        similarity=None,
):  
    if type(st_model) is str:
        st_model = SentenceTransformer(st_model)
    key_emb = st_model.encode(key, convert_to_tensor=True)
    query_emb = st_model.encode(query)
    similarity = st_model.similarity(key_emb, query_emb)
    return similarity.item()

def set_scores(
    con,
    table_name,
    query,
    sentence_transformers_model=SentenceTransformer("all-mpnet-base-v2"),
    similaity=None,
):
    model = SentenceTransformer(sentence_transformers_model)
    df_kw_link = pd.read_sql(f"SELECT name, keywords, short_summary FROM {table_name};", con)
    kwds_emb = model.encode(df_kw_link.keywords, convert_to_tensor=True)
    query_emb = model.encode(query)
    similarities = model.similarity(kwds_emb, query_emb)
    df_kw_link["score"] = similarities[:, 0]
    cursor = con.cursor()
    query = f"UPDATE {table_name} SET score = ? WHERE name = ?"
    values = list(df_kw_link[["score", "name"]].itertuples(index=False, name=None))
    cursor.executemany(query, values)
    con.commit()
