import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch



def beautify_col(col: str) -> str:
    return col.capitalize().replace("_", " ")

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
    barWidth = 0.3
    br1 = np.arange(16) 
    br2 = [i + barWidth for i in br1]
    #br3 = [x + barWidth for x in br2]
    #binsCyc = [1, 11, 21, 51, 171]
    binsCyc = range(1, 172, 10)
    binsCod = np.arange(0, 1.1, 0.1)
    cols = ["cyclomatic_complexity", "code_coverage"]
    bins = [binsCyc, binsCod]
    colors = ["orange", "blue"]
    df_new = df.copy()
    #df_new.loc[df_new["compilation_iteration"].isna(), "compilation_iteration"] = 16
    df_new["compilation_iteration"] = df_new["compilation_iteration"].fillna(16)
    df_new["testing_iteration"] = df_new["testing_iteration"].fillna(16)
    df_new["cyclomatic_complexity"] = df_new["cyclomatic_complexity"].fillna(max(df_new["cyclomatic_complexity"]))
    df_new["code_coverage"] = df_new["code_coverage"].fillna(0)
    legend_elements = [
        Patch(facecolor="tab:olive", label="Compilation"),
        Patch(facecolor="tab:green", label="Testing"),
    ]
    median_style = {"color": "tab:red"}

    if False:
        # Global plots
        ## Boxplot
        bp = plt.boxplot(
            [df_new["compilation_iteration"].dropna(), df_new["testing_iteration"].dropna()], 
            tick_labels=["Compilation", "Testing"], 
            patch_artist=True,
            medianprops=median_style
        )
        bp["boxes"][0].set_facecolor("tab:olive")
        bp["boxes"][1].set_facecolor("tab:green")
        plt.title("Compilation and Testing")
        plt.ylabel("Iterations")
        plt.legend(handles=legend_elements, loc="best")
        plt.show()
        ## Barplot
        cmpl = [ df_new.loc[df_new["compilation_iteration"] == i + 1, "compilation_iteration"].count() for i in br1 ] 
        test = [ df_new.loc[df_new["testing_iteration"] == i + 1, "testing_iteration"].count() for i in br1] 
        plt.bar(br1, cmpl, color="tab:olive", edgecolor="grey", width=barWidth, label="Compilation") 
        plt.bar(br2, test, color="tab:green", edgecolor="grey", width=barWidth, label="Testing") 
        plt.title("Compilation and Testing")
        plt.xlabel("Iterations")
        plt.xticks([i + (barWidth/2) for i in br1], br1 + 1)
        plt.legend(loc="best")
        plt.show()
        ## Boxplot
        col = "cyclomatic_complexity"
        bp = plt.boxplot(df_new[col].dropna(), tick_labels=[""], patch_artist=True, medianprops=median_style)
        bp["boxes"][0].set_facecolor("tab:orange")
        plt.title(beautify_col(col))
        plt.show()
        ## Histogram
        plt.hist(df_new[col].dropna(), color="tab:orange", edgecolor="orange", bins=binsCyc)
        plt.title(beautify_col(col))
        plt.xticks(binsCyc) 
        plt.show()
        ## Boxplot
        col = "code_coverage"
        plt.boxplot(df_new[col].dropna(), tick_labels=[""], patch_artist=True, medianprops=median_style)
        bp["boxes"][0].set_facecolor("tab:blue")
        plt.title(beautify_col(col))
        plt.show()
        ## Histogram
        plt.hist(df_new[col].dropna(), color="tab:blue", edgecolor="blue", bins=binsCod)
        plt.title(beautify_col(col))
        plt.xticks(binsCod)
        plt.show()
    
    #raise SystemExit
        
    # LLM plots
    llms = ["Anthropic", "Google", "OpenAI"]
    df_anth = df_new[df_new["llm"] == "anthropic"]
    df_goog = df_new[df_new["llm"] == "google"]
    df_open = df_new[df_new["llm"] == "openai"]
    ## Boxplot
    bp = plt.boxplot(
        [ 
            df_anth["compilation_iteration"].dropna(), 
            df_anth["testing_iteration"].dropna(), 
            df_goog["compilation_iteration"].dropna(),
            df_goog["testing_iteration"].dropna(),
            df_open["compilation_iteration"].dropna(),
            df_open["testing_iteration"].dropna()
        ], 
        positions=[1, 2, 4, 5, 7, 8], 
        patch_artist=True,
        medianprops=median_style
    )
    # Set legend
    for i in [0, 2, 4]:
        bp["boxes"][i].set_facecolor("tab:olive")
    for i in [1, 3, 5]:
        bp["boxes"][i].set_facecolor("tab:green")
    # Tick positions = centers of each pair
    tick_positions = [1.5, 4.5, 7.5]
    plt.xticks(tick_positions, llms)
    plt.title("Compilation and Testing")
    plt.ylabel("Iterations")
    plt.legend(handles=legend_elements, loc="best")
    plt.show()
    ## Barplot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))  # 3 rows, 1 column
    data = [
        df_anth,
        df_goog,
        df_open
    ]
    for i in range(len(data)):
        df = data[i]
        cmpl = [ df.loc[df["compilation_iteration"] == j + 1, "compilation_iteration"].count() for j in br1 ] 
        test = [ df.loc[df["testing_iteration"] == j + 1, "testing_iteration"].count() for j in br1] 
        axes[i].bar(br1, cmpl, color="tab:olive", edgecolor="grey", width=barWidth, label="Compilation") 
        axes[i].bar(br2, test, color="tab:green", edgecolor="grey", width=barWidth, label="Testing") 
        axes[i].set_title(llms[i])
        #axes[i].set_xlabel("Iterations")
        axes[i].set_xticks([j + (barWidth/2) for j in br1], br1 + 1)
        axes[i].set_ylim(0, 160)
        axes[i].legend(handles=legend_elements, loc="upper center")
    axes[len(data) - 1].set_xlabel("Iterations")
    plt.tight_layout()
    plt.show()

    ylims = [80, 160]
    for i in range(len(cols)):
        col = cols[i]
        ticks = bins[i]
        color = colors[i]
        ## Boxplot
        data = [
            df_anth[col].dropna(), 
            df_goog[col].dropna(),
            df_open[col].dropna()
        ]
        bp = plt.boxplot(
            data,
            tick_labels=llms,
            patch_artist=True, 
            medianprops=median_style
        )
        for j in range(len(data)):
            bp["boxes"][j].set_facecolor(f"tab:{color}")
        plt.title(beautify_col(col))
        plt.show()
        ## Histogram
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))  # 3 rows, 1 column
        for j in range(len(data)):
            axes[j].hist(data[j], color=f"tab:{color}", edgecolor=color, bins=ticks)
            axes[j].set_title(llms[j])
            axes[j].set_xticks(ticks)
            axes[j].set_ylim(0, ylims[i])
        axes[len(data) - 1].set_xlabel(beautify_col(col))
        plt.tight_layout()
        plt.show()

    #raise SystemExit

    # Architecture plots
    archs = ["Single-agent", "Multi-agent"]
    df_sa = df_new[df_new["type"] == "single_agent"]
    df_ma = df_new[df_new["type"] == "multi_agent"]
    data = [ 
        df_sa["compilation_iteration"].dropna(), 
        df_sa["testing_iteration"].dropna(), 
        df_ma["compilation_iteration"].dropna(),
        df_ma["testing_iteration"].dropna()
    ]
    ## Boxplot
    bp = plt.boxplot(
        data, 
        positions=[1, 2, 4, 5], 
        patch_artist=True,
        medianprops=median_style
    )
    for i in [0, 2]:
        bp["boxes"][i].set_facecolor("tab:olive")
    for i in [1, 3]:
        bp["boxes"][i].set_facecolor("tab:green")
    tick_positions = [1.5, 4.5]
    plt.xticks(tick_positions, archs)
    plt.title("Compilation and Testing")
    plt.ylabel("Iterations")
    plt.legend(handles=legend_elements, loc="best")
    plt.show()
    ## Barplot
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    data = [
        df_sa,
        df_ma
    ]
    for i in range(len(data)):
        df = data[i]
        cmpl = [ df.loc[df["compilation_iteration"] == j + 1, "compilation_iteration"].count() for j in br1 ] 
        test = [ df.loc[df["testing_iteration"] == j + 1, "testing_iteration"].count() for j in br1] 
        axes[i].bar(br1, cmpl, color="tab:olive", edgecolor="grey", width=barWidth, label="Compilation") 
        axes[i].bar(br2, test, color="tab:green", edgecolor="grey", width=barWidth, label="Testing") 
        axes[i].set_title(archs[i])
        #axes[i].set_xlabel("Iterations")
        axes[i].set_xticks([j + (barWidth/2) for j in br1], br1 + 1)
        axes[i].set_ylim(0, 220)
        axes[i].legend(handles=legend_elements, loc="upper center")
    axes[len(data) - 1].set_xlabel("Iterations")
    plt.tight_layout()
    plt.show()

    ylims = [120, 120]
    for i in range(len(cols)):
        col = cols[i]
        ticks = bins[i]
        color = colors[i]
        ## Boxplot
        data = [
            df_sa[col].dropna(), 
            df_ma[col].dropna()
        ]
        bp = plt.boxplot(
            data,
            tick_labels=archs,
            patch_artist=True, 
            medianprops=median_style
        )
        for j in range(len(data)):
            bp["boxes"][j].set_facecolor(f"tab:{color}")
        plt.title(beautify_col(col))
        plt.show()
        ## Histogram
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        for j in range(len(data)):
            axes[j].hist(data[j], color=f"tab:{color}", edgecolor=color, bins=ticks)
            axes[j].set_title(archs[j])
            axes[j].set_xticks(ticks)
            axes[j].set_ylim(0, ylims[i])
        axes[len(data) - 1].set_xlabel(beautify_col(col))
        plt.tight_layout()
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
