import pandas as pd
import logging
import logging
import sqlite3
from pathlib import Path
from argparse import ArgumentParser
import json
from .augsearch import (
    prepare_table,
    get_links,
    get_keywords_and_summaries,
    get_llmbot_connector,
    set_scores,
    load_json
)

logger = logging.getLogger()


def main():
    parser = ArgumentParser()
    parser.add_argument("--names", required=True)
    parser.add_argument("--db", default="sci_names.db")
    parser.add_argument("--llm_params", default=None)
    parser.add_argument("--num_websites", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--table_name", default="out")
    parser.add_argument(
        "--score_query",
        default="reinforcement learning,  deep learning, computer vision, healthcare, biology",
    )
    parser.add_argument("--if_exists", default="skip", choices=["skip", "replace", "append", "fail"])
    parser.add_argument("--out_csv", default="")
    args = parser.parse_args()

    

    if args.llm_params is None:
        llm_params = {
            "which": "ollama",
            "host" : f"http://localhost:11434",
            "headers": {"x-some-header": "some-value"},
            "model": "llama3.2:1b",
            "context_window": 8000
        }
    else:
        llm_params = load_json(args.llm_params)

    llm_connector = get_llmbot_connector(llm_params)

    database_filen = args.db if args.db.endswith(".db") else  args.db + ".db"
    con = sqlite3.connect(args.db)
    df_path = Path(args.names)
    assert str(df_path).endswith(".csv"), f"Expected csv datasheet, got: {df_path.name}"
    df_names = pd.read_csv(df_path)
    # num_websites = 3

    if args.table_name == "":
        table_name = df_path.name.replace(".csv", "")
    else:
        table_name = args.table_name

    prepare_table(df_names, con=con, table_name=table_name, num_websites=args.num_websites, if_exists=args.if_exists)
    get_links(con=con, table_name=table_name, label=args.label, num_websites=args.num_websites)
    get_keywords_and_summaries(
        con=con,
        table_name=table_name,
        num_websites=args.num_websites,
        llm_connector = llm_connector,
        score_query=args.score_query
    )

    # logger.info(f"Setting scores for the query: {args.score_query}")

    # set_scores(con, table_name, query=args.score_query)

    if args.out_csv:
        df_out = pd.read_sql(f"SELECT * FROM {table_name}", con=con)
        df_out = df_out.sort_values(by="score", ascending=False)
        df_out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    # main()
    print("Hello world!")
