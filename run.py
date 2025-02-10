import pandas as pd
import logging
import logging
import sqlite3
from pathlib import Path
from argparse import ArgumentParser
from sci_search import (
    prepare_table,
    get_links,
    get_keywords_and_best_links,
    get_ollama_client,
    set_scores,
)

logger = logging.getLogger()


def main():
    parser = ArgumentParser()
    parser.add_argument("--names", required=True)
    parser.add_argument("--db", default="sci_names.db")
    parser.add_argument("--ollama_model", default="llama3.2:1b")
    parser.add_argument("--context_window", default=8000)
    parser.add_argument("--num_websites", default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--table_name", default="")
    parser.add_argument(
        "--score_query",
        default="reinforcement learning,  deep learning, computer vision, healthcare, biology",
    )
    parser.add_argument(
        "--if_exists", default="skip", choices=["skip", "replace", "append", "fail"]
    )
    parser.add_argument("--out_csv", default="")
    args = parser.parse_args()

    ollama_client = get_ollama_client()

    con = sqlite3.connect(args.db)
    df_path = Path(args.names)
    assert str(df_path).endswith(".csv"), f"Expected csv datasheet, got: {df_path.name}"
    df_names = pd.read_csv(df_path)
    num_websites = 3

    if args.table_name == "":
        table_name = df_path.name.replace(".csv", "")
    else:
        table_name = args.table_name

    prepare_table(
        df_names,
        con=con,
        table_name=table_name,
        num_websites=num_websites,
        if_exists=args.if_exists,
    )
    get_links(
        con=con, table_name=table_name, label=args.label, num_websites=num_websites
    )
    get_keywords_and_best_links(
        con=con,
        table_name=table_name,
        num_websites=num_websites,
        model_name=args.ollama_model,
        ollama_client=ollama_client,
        context_window=args.context_window,
    )

    logger.info(f"Setting scores for the query: {args.score_query}")

    set_scores(con, table_name, query=args.score_query)

    if args.out_csv:
        df_out = pd.read_sql(f"SELECT * FROM {table_name}", con=con)
        df_out = df_out.sort_values(by="score", ascending=False)
        df_out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
