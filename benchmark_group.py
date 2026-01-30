import pandas as pd
import matplotlib.pyplot as plt
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

    # save
    df.to_csv(benchmarks_dir / f"benchmarks_new.csv", encoding="utf-8", sep=",", na_rep="", float_format="%.4f")

    # set index and check it
    df["index"] = df["n"].astype("string") + "|" + df["type"] + "|" + df["file_format"] + "|" + df["llm"]
    df.set_index("index", inplace=True, verify_integrity=True)
    
    df.info()
    print(df)

    # general
    cols = ["cyclomatic_complexity", "code_coverage"]
    llms = ["anthropic", "google", "openai"]

    # global plots
    data = [ 
        df["compilation_iteration"].dropna(), 
        df["testing_iteration"].dropna() 
    ]
    plt.boxplot(data, labels=["Compilation", "Testing"])
    plt.ylabel("Iterations")
    plt.title("Iterations by phase")
    plt.show()
    for col in cols:
        plt.boxplot(df[col].dropna())
        col_out = col.capitalize().replace("_", " ")
        plt.ylabel(col_out)
        plt.title(f"Distribution of {col_out}")
        plt.show()
    
    # LLM plots
    data = [ 
        df.loc[df["llm"] == "anthropic", "compilation_iteration"].dropna(), 
        df.loc[df["llm"] == "anthropic", "testing_iteration"].dropna(), 
        df.loc[df["llm"] == "google", "compilation_iteration"].dropna(),
        df.loc[df["llm"] == "google", "testing_iteration"].dropna(),
        df.loc[df["llm"] == "openai", "compilation_iteration"].dropna(),
        df.loc[df["llm"] == "openai", "testing_iteration"].dropna()
    ]
    plt.boxplot(data, 
                labels=["Compilation anthropic", "Testing anthropic", "Compilation google", "Testing google", "Compilation openai", "Testing openai"], 
                positions=[1, 2, 4, 5, 7, 8])
    plt.ylabel("Iterations")
    plt.title("Iterations by LLM and phase")
    plt.show()
    
    for col in cols:
        data = [ 
            df.loc[df["llm"] == "anthropic", col].dropna(), 
            df.loc[df["llm"] == "google", col].dropna(),
            df.loc[df["llm"] == "openai", col].dropna() 
        ]
        plt.boxplot(data, labels=llms)
        col_out = col.capitalize().replace("_", " ")
        plt.title(f"Distribution of {col_out}")
        plt.show()

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
            std_compilation_iteration=("compilation_iteration", "std"),

            avg_testing_iteration=("testing_iteration", "mean"),
            std_testing_iteration=("testing_iteration", "std"),

            #avg_validation_iteration=("validation_iteration", "mean"),
            #std_validation_iteration=("validation_iteration", "std"),

            avg_cyclomatic_complexity=("cyclomatic_complexity", "mean"),
            std_cyclomatic_complexity=("cyclomatic_complexity", "std"),

            avg_code_coverage=("code_coverage", "mean"),
            std_code_coverage=("code_coverage", "std"),

            avg_end_seconds=("end_seconds", "mean"),
            std_end_seconds=("end_seconds", "std")
        )
        # group rate calculation
        df_group["compilation_rate"] = df_group["cnt_compiled"] / df_group["cnt_all"]
        df_group["testing_rate_abs"] = df_group["cnt_testing"] / df_group["cnt_all"]
        df_group["testing_rate_rel"] = df_group["cnt_testing"] / df_group["cnt_compiled"]
        #df_group["validation_rate_abs"] = df_group["cnt_validated"] / df_group["cnt_all"]
        #df_group["validation_rate_rel"] = df_group["cnt_validated"] / df_group["cnt_testing"]
        # drop stats not useful anymore
        df_group.drop(columns=["cnt_compiled", "cnt_testing", "cnt_validated", "cnt_all"], inplace=True)
        # more compact columns names
        df_group.rename(columns={
            "avg_compilation_iteration": "avg_cmpl_iter", 
            "std_compilation_iteration": "std_cmpl_iter",
            "avg_testing_iteration": "avg_test_iter", 
            "std_testing_iteration": "std_test_iter",
            #"avg_validation_iteration": "avg_vald_iter", 
            #"std_validation_iteration": "std_vald_iter",
            "avg_end_seconds": "avg_end_time", 
            "std_end_seconds": "std_end_time",
            "compilation_rate": "cmpl_rate",
            "testing_rate_abs": "test_rate_abs",
            "testing_rate_rel": "test_rate_rel",
            #"validation_rate_abs": "vald_rate_abs",
            #"validation_rate_rel": "vald_rate_rel"
        }, inplace=True)
        # print
        df_group.info()
        print(df_group)
        # save
        df_group.to_csv(benchmarks_dir / f"benchmarks_{name}.csv", encoding="utf-8", sep=",", na_rep="", float_format="%.4f")
