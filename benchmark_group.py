import pandas as pd
from pathlib import Path



if __name__ == "__main__":
    # read
    benchmarks_dir = Path("benchmark/all")
    df = pd.read_csv(benchmarks_dir / "benchmarks.csv", encoding="utf-8", sep=",")
    
    # casts
    df["type"] = df["type"].astype("string")
    df["file_format"] = df["file_format"].astype("string")
    df["llm"] = df["llm"].astype("string")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["compilation_time"] = pd.to_datetime(df["compilation_time"])
    df["testing_time"] = pd.to_datetime(df["testing_time"])
    df["validation_time"] = pd.to_datetime(df["validation_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # edits
    df["code_coverage"] /= 100
    df.loc[df["type"] == "zero_shot", "type"] = "single_agent"
    df["compilation_seconds"] = (df["compilation_time"] - df["start_time"]).dt.total_seconds().astype("float64")
    df["testing_seconds"] = (df["testing_time"] - df["start_time"]).dt.total_seconds().astype("float64")
    df["validation_seconds"] = (df["validation_time"] - df["start_time"]).dt.total_seconds().astype("float64")
    df["end_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds().astype("float64")

    # drops
    df.drop(columns=[
        # useless
        "best_parser_folder", "testing_rate",
        # not useful anymore
        "start_time", "compilation_time", "testing_time", "validation_time", "end_time"
    ], inplace=True)

    # set index and check it
    df["index"] = df["n"].astype("string") + "|" + df["type"] + "|" + df["file_format"] + "|" + df["llm"]
    df.set_index("index", inplace=True, verify_integrity=True)
    
    df.info()
    print(df)

    #df2 = df[(df["type"] == "single_agent") & (df["llm"] == "openai")]
    #df2 = df2[["file_format", "compilation_iteration", "testing_iteration", "best_parser_folder"]]
    #df2["diff"] = df2["testing_iteration"] - df2["compilation_iteration"]

    #for r in df2.itertuples():
    #    print(r)
    
    #df2 = df2[df2["diff"] > 1]
    #print(df2)

    #for r in df2.itertuples():
    #    print(r.best_parser_folder)

    #raise SystemExit

    groups = {
        "tfl": ["type", "file_format", "llm"],
        "tl": ["type", "llm"],
        "t": ["type"],
        "l": ["llm"]
    }

    for name, group in groups.items():
        # group aggregation
        df_group = df.copy().groupby(group).agg(
            cnt_all=("n", "count"),
            cnt_compiled=("compilation_iteration", "count"),
            cnt_testing=("testing_iteration", "count"),
            cnt_validated=("validation_iteration", "count"),
            avg_compilation_iteration=("compilation_iteration", "mean"),
            avg_testing_iteration=("testing_iteration", "mean"),
            avg_validation_iteration=("validation_iteration", "mean"),
            avg_cyclomatic_complexity=("cyclomatic_complexity", "mean"),
            avg_code_coverage=("code_coverage", "mean"),
            # not interesting stats
            #avg_compilation_seconds=("compilation_seconds", "mean"),
            #avg_testing_seconds=("testing_seconds", "mean"),
            #avg_validation_seconds=("validation_seconds", "mean"),
            avg_end_seconds=("end_seconds", "mean")
        )
        # group rate calculation
        df_group["compilation_rate"] = df_group["cnt_compiled"] / df_group["cnt_all"]
        df_group["testing_rate_abs"] = df_group["cnt_testing"] / df_group["cnt_all"]
        df_group["testing_rate_rel"] = df_group["cnt_testing"] / df_group["cnt_compiled"]
        df_group["validation_rate_abs"] = df_group["cnt_validated"] / df_group["cnt_all"]
        df_group["validation_rate_rel"] = df_group["cnt_validated"] / df_group["cnt_testing"]
        # drop stats not useful anymore
        df_group.drop(columns=["cnt_compiled", "cnt_testing", "cnt_validated", "cnt_all"], inplace=True)
        # print
        df_group.info()
        print(df_group)
        # save
        df_group.to_csv(benchmarks_dir / f"benchmarks_{name}.csv", encoding="utf-8", sep=",", na_rep="", float_format="%.4f")
