import pandas as pd
from tqdm.auto import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from googlesearch import search
from bs4 import BeautifulSoup
from ollama import Client
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup, Comment
import unicodedata
import html
import re
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()


def get_ollama_client(port="11434"):
    return Client(
        host=f"http://localhost:{port}", headers={"x-some-header": "some-value"}
    )


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
    for tag in soup.find_all(
        attrs={
            "style": lambda v: v
            and (
                "display: none" in v.lower()
                or "visibility: hidden" in v.lower()
                or "opacity: 0" in v.lower()
            )
        }
    ):
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
    tokenizer=AutoTokenizer.from_pretrained(
        "google-bert/bert-base-cased", verbose=False
    ),
):
    tokens = tokenizer.tokenize(text, verbose=False)
    num_tokens = len(tokens)

    if hasattr(tokenizer, "name_or_path"):
        tokenizer_name = tokenizer.name_or_path
    else:
        tokenizer_name = "Unknown"
    if num_tokens > context_window:
        logger.warning(
            f"The text might be too long. Num tokens:{num_tokens}. Context window: {context_window}. Tokenizer: {tokenizer_name}"
        )
        return False, num_tokens
    else:
        logger.info(
            f"The text length most likely is fine. Num tokens:{num_tokens}. Context window {context_window}. Tokenizer: {tokenizer_name}"
        )
        return True, num_tokens


def get_site_contents(url, lines_set=True):
    response = requests.get(url)
    if response.status_code != 200:
        return [f"Got response with status code: {response.status_code}"], []

    soup = BeautifulSoup(response.text)

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


def get_links(name, label="EPFL", num_websites=3):
    search_query = f"{name} + {label}"
    results = list(search(search_query, stop=num_websites))

    links_content_list = []
    for url in results:
        lines_url, links_url = get_site_contents(url)
        links_content_list.append((url, "\n".join(lines_url)))
        # lines_dict[url] = '\n'.join(lines_url)
    return links_content_list


def compose_query(links_dict, name):
    num_websites = len(links_dict)
    query = f"""
    These are several links ({num_websites} in total) and the texts from the corresponding websites. LEt's call the MAIN links
    All the websites are presumably related to a researcher, their name is {name}.
    The texts may have some html or other web garbage:\n
    """

    for link_n, (link, text) in enumerate(links_dict.items()):
        link_query = f"MAIN Link #{link_n}: {link}:\n\n<<<\n{text}\n>>>\n\n"
        query += link_query

    query += f"""
    Please, figure out what their scientific interests are. Provide the answer as 5 keywords.
    From all the provided links choose the one which was the most helpful.  Choose one of MAIN links, which are: {links_dict.keys()}
    After all the keywords append the number which represents your cumulative confidence in the total reply.
    The confidence should be from 0 to 10  -- 0 is no confidence, 10 is totally confident. 
    Lower confidence if the text seems not to be related to a researcher or has any other unexpected anomalies.
    Don't include anything but the kewords and the confidence number. Put each item on a separate line.
    The final answer must consitist of 7 lines in total: 5 lines with the keywords, one line for the most useful link (MAIN link URL only) and one line with the cumulative confidence number. 
    No empty lines in between. Don't separate the link or score with any extra empty lines. The final format is really important.
    """

    return query


def ask_ollama(query, model, context_window=128000, client=get_ollama_client()):
    check_num_tokens(text=query, context_window=context_window)
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        options={"num_ctx": context_window},
    )

    return response


def clean_with_ollama(text, model, client, context_window=128000):
    query = f"""
    This is the text from a website related to a researcher.  
    Try figure out their reserach interests. Filter out the garbage, but keep all the relevant to theire research topics, interests, publications and so on.
    The text:
    <<<
    {text}
    >>>
    """
    # print(query)

    response = ask_ollama(
        query=query, model=model, client=client, context_window=context_window
    )
    return response.message.content


def summarize_with_ollama(links_contents, name, model, client, context_window):
    num_websites = len(links_contents)
    links = [x[0] for x in links_contents]
    query = f"""
    These are several websites summaries ({num_websites} and the corresponding links. 
    All the websites are presumably related to a researcher, their name is {name}.
    The links and texts are the following:

    """

    for link_n, (link, text) in enumerate(links_contents):
        link_query = f"\nLink #{link_n}: {link}:\n\n<<<\n{text}\n>>>\n\n"
        query += link_query

    query += f"""
    Please, figure out what their scientific interests are. Provide the answer as 5 keywords.
    From all the provided links choose the one which was the most helpful.  Choose one of original links, which are: {links} and 
    The final answer must consist of 6 lines in total: 5 lines with the keywords, one line for the most useful link (URL only)
    Don't put any numbers titles or anything else, the answer must be palin gkeywords and a link.
    No empty lines in between. Don't separate the link  with any extra empty lines. The final format is really important.
    """
    response = ask_ollama(
        query=query, model=model, client=client, context_window=context_window
    )
    return response.message.content


def fix_format_with_ollama(inputs, name, model, client, context_window):
    query = f"""
    This text should contatain 5 keywords and a link. Put them in the following format:
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
    response = ask_ollama(
        query=query, model=model, client=client, context_window=context_window
    )
    return response.message.content


def parse_summary(summary):
    split = summary.strip().split("</think>")[-1].split("\n")
    split = [x for x in split if len(x)]
    try:
        assert len(split) == 6
        kws = ", ".join(split[:5])
        link = split[5]
        return kws, link

    except:
        logger.warning("Failed to parse keywords in line by line mode")
    try:
        assert len(split) == 2
        kws = split[0]
        link = split[1]
        return kws, link
    except:
        logger.warning("Failed to parse keywords in 2 line mode")

    kws = summary.replace("\n", ", ")
    link = ""
    logger.warning(f"Failed to parse the formatting:\n {summary}")
    return kws, link


def prepare_table(inputs_df, con, table_name, num_websites, if_exists="skip"):
    tables_df = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table';", con)
    exists = table_name in list(tables_df.name)

    if exists and if_exists == "skip":
        print("Table exists, doing nothing")
    else:
        if exists and "if_exists" == "replace":
            logger.warning(f"Repacing already present table: {table_name}")
        fields = ["name", "score", "keywords", "best_link"] + [
            f"link_{x}" for x in range(num_websites)
        ]
        fields_empty_on_init = [x for x in fields if x not in ["name", "score"]]
        df = pd.DataFrame(columns=fields)

        df.name = inputs_df.name
        df = df.assign(**{x: "" for x in fields_empty_on_init})
        df.to_sql(
            name=table_name,
            con=con,
            if_exists="fail" if if_exists == "skip" else if_exists,
        )


def get_links(con, table_name, label, num_websites):
    # names_no_links, _ = get_names_to_process(df_names, con=con, table_name=table_name, column="link_0")
    names_no_links = pd.read_sql(
        f"SELECT name FROM {table_name} WHERE link_0 == '';", con=con
    ).name
    cur = con.cursor()
    for name in tqdm(names_no_links, desc="Getting the links"):
        search_query = f"{name} + {label}"
        results = list(search(search_query, stop=num_websites))
        clause_line = ", ".join([f"link_{i} = ?" for i in range(len(results))])
        cur.execute(
            f"UPDATE {table_name} SET {clause_line} WHERE name = ?",
            tuple(results) + (name,),
        )
        con.commit()
    return names_no_links


def get_keywords_and_best_links(
    con, table_name, num_websites, model_name, ollama_client, context_window
):
    fields = ["name"] + [f"link_{x}" for x in range(num_websites)]
    fields = ", ".join(fields)
    names_links_no_kws = pd.read_sql(
        f"SELECT {fields} FROM {table_name} WHERE keywords == '' AND link_0 != ''",
        con=con,
    )
    cursor = con.cursor()
    summaries = []
    for idr, row in tqdm(
        names_links_no_kws.iterrows(),
        total=len(names_links_no_kws),
        desc="Getting keywords and best link:",
    ):
        links_contents = []
        for website_i in range(num_websites):
            url = row[f"link_{website_i}"]
            contents, _ = get_site_contents(url)
            contents_cleaned = clean_html_text("\n".join(contents))
            ollama_cleaned_content = clean_with_ollama(
                contents_cleaned,
                model=model_name,
                client=ollama_client,
                context_window=context_window,
            )
            links_contents.append((url, ollama_cleaned_content))
        name = row["name"]
        summary = summarize_with_ollama(
            links_contents,
            name=name,
            model=model_name,
            client=ollama_client,
            context_window=context_window,
        )
        summaries.append(summary)
        kws, best_link = parse_summary(summary)
        # print(f"UPDATE {table_name} SET keywords = '{kws}', best_link = '{best_link}' WHERE name = '{name}';")
        cursor.execute(
            f"UPDATE {table_name} SET keywords = ?, best_link = ? WHERE name = ?;",
            (kws, best_link, name),
        )
        con.commit()


def set_scores(
    con,
    table_name,
    query,
    sentence_transformers_model="all-mpnet-base-v2",
    similaity=None,
):
    model = SentenceTransformer(sentence_transformers_model)
    df_kw_link = pd.read_sql(
        f"SELECT name, keywords, best_link FROM {table_name};", con
    )
    kwds_emb = model.encode(df_kw_link.keywords, convert_to_tensor=True)
    query_emb = model.encode(query)
    similarities = model.similarity(kwds_emb, query_emb)
    df_kw_link["score"] = similarities[:, 0]
    cursor = con.cursor()
    query = f"UPDATE {table_name} SET score = ? WHERE name = ?"
    values = list(df_kw_link[["score", "name"]].itertuples(index=False, name=None))
    cursor.executemany(query, values)
    con.commit()
