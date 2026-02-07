import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch


REPLACE_NA = False
SHOW_PLOTS = True
SHOW_BARPLOTS = False

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

    df_new = df.copy()
    if REPLACE_NA:
        #df_new.loc[df_new["compilation_iteration"].isna(), "compilation_iteration"] = 16
        df_new["compilation_iteration"] = df_new["compilation_iteration"].fillna(16)
        df_new["testing_iteration"] = df_new["testing_iteration"].fillna(16)
        df_new["cyclomatic_complexity"] = df_new["cyclomatic_complexity"].fillna(max(df_new["cyclomatic_complexity"]))
        df_new["code_coverage"] = df_new["code_coverage"].fillna(0)

    if SHOW_PLOTS:
        # Initialization
        barWidth = 0.25
        br1 = np.arange(16) 
        br2 = [i + barWidth for i in br1]
        br3 = [x + barWidth for x in br2]
        binsCyc = [1, 10, 20, 50]
        #binsCyc = [1, 11, 21, 51, 171]
        #binsCyc = range(1, 172, 10)
        binsCod = np.arange(0, 1.1, 0.1)
        bins = [binsCyc, binsCod]
        cols = ["cyclomatic_complexity", "code_coverage"]
        it_cols = ["compilation_iteration", "testing_iteration"]
        legend_steps = [
            Patch(facecolor="olive", label="Compilation", alpha=0.8),
            Patch(facecolor="darkgreen", label="Testing", alpha=0.8),
        ]
        legend_llms = [
            Patch(facecolor="blue", label="Anthropic", alpha=0.4),
            Patch(facecolor="red", label="Google", alpha=0.4),
            Patch(facecolor="green", label="OpenAI", alpha=0.4),
        ]
        legend_archs = [
            Patch(facecolor="orange", label="Single-agent", alpha=0.6),
            Patch(facecolor="purple", label="Multi-agent", alpha=0.6),
        ]
        median_style = {"color": "tab:red", "linewidth": 1.5}
            
        # LLM plots
        llms = ["Anthropic", "Google", "OpenAI"]
        df_anth = df_new[df_new["llm"] == "anthropic"]
        df_goog = df_new[df_new["llm"] == "google"]
        df_open = df_new[df_new["llm"] == "openai"]
        ## Boxplot
        data = [ 
            df_anth["compilation_iteration"].dropna(), 
            df_anth["testing_iteration"].dropna(), 
            df_goog["compilation_iteration"].dropna(),
            df_goog["testing_iteration"].dropna(),
            df_open["compilation_iteration"].dropna(),
            df_open["testing_iteration"].dropna()
        ]
        bp = plt.boxplot(
            data, 
            positions=[1, 2, 4, 5, 7, 8], 
            patch_artist=True,
            medianprops=median_style
        )
        # Set legend
        for i in range(len(data)):
            c = legend_steps[0 if (i % 2) == 0 else 1].get_facecolor()
            bp["boxes"][i].set_facecolor(c)
        # Tick positions = centers of each pair
        tick_positions = [1.5, 4.5, 7.5]
        plt.xticks(tick_positions, llms)
        plt.title("Compilation and Testing")
        plt.ylabel("Iterations")
        plt.legend(handles=legend_steps, loc="best")
        plt.show()
        if SHOW_BARPLOTS:
            ## Barplot
            fig, axes = plt.subplots(2, 1, figsize=(12, 9))  # 2 rows, 1 column
            for i, it_col in enumerate(it_cols):
                df_anth_col = [ df_anth.loc[df_anth[it_col] == j + 1, it_col].count() for j in br1 ]
                df_goog_col = [ df_goog.loc[df_goog[it_col] == j + 1, it_col].count() for j in br1 ] 
                df_open_col = [ df_open.loc[df_open[it_col] == j + 1, it_col].count() for j in br1 ] 
                axes[i].bar(br1, df_anth_col, color=legend_llms[0].get_facecolor(), edgecolor="grey", width=barWidth, label="Anthropic") 
                axes[i].bar(br2, df_goog_col, color=legend_llms[1].get_facecolor(), edgecolor="grey", width=barWidth, label="Google") 
                axes[i].bar(br3, df_open_col, color=legend_llms[2].get_facecolor(), edgecolor="grey", width=barWidth, label="OpenAI") 
                axes[i].set_title(beautify_col(it_col))
                axes[i].set_xticks([j + barWidth for j in br1], br1 + 1)
                axes[i].set_ylim(0, 200)
                axes[i].legend(handles=legend_llms, loc="upper center")
            axes[-1].set_xlabel("Iterations")
            plt.tight_layout()
            plt.show()

        for i in range(len(cols)):
            col = cols[i]
            ticks = bins[i]
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
                c = legend_llms[j].get_facecolor()
                bp["boxes"][j].set_facecolor(c)
            plt.title(beautify_col(col))
            plt.show()
            ## Histogram
            for j in range(len(data)):
                c = legend_llms[j].get_facecolor()
                sns.histplot(data[j], color=c, edgecolor=c, label=llms[j], bins=ticks, kde=True)
            plt.title(f"{beautify_col(col)} (KDE)")
            plt.xlabel("")
            plt.xticks(ticks)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Architecture plots
        archs = ["Single-agent", "Multi-agent"]
        df_sa = df_new[df_new["type"] == "single_agent"]
        df_ma = df_new[df_new["type"] == "multi_agent"]
        ## Boxplot
        data = [
            df_sa["compilation_iteration"].dropna(), 
            df_sa["testing_iteration"].dropna(), 
            df_ma["compilation_iteration"].dropna(),
            df_ma["testing_iteration"].dropna()
        ]
        bp = plt.boxplot(
            data, 
            positions=[1, 2, 4, 5], 
            patch_artist=True,
            medianprops=median_style
        )
        for i in range(len(data)):
            c = legend_steps[0 if (i % 2) == 0 else 1].get_facecolor()
            bp["boxes"][i].set_facecolor(c)
        tick_positions = [1.5, 4.5]
        plt.xticks(tick_positions, archs)
        plt.title("Compilation and Testing")
        plt.ylabel("Iterations")
        plt.legend(handles=legend_steps, loc="best")
        plt.show()
        if SHOW_BARPLOTS:
            ## Barplot
            fig, axes = plt.subplots(2, 1, figsize=(12, 9))
            for i, it_col in enumerate(it_cols):
                df_sa_col = [ df_sa.loc[df_sa[it_col] == j + 1, it_col].count() for j in br1 ]
                df_ma_col = [ df_ma.loc[df_ma[it_col] == j + 1, it_col].count() for j in br1 ] 
                axes[i].bar(br1, df_sa_col, color=legend_archs[0].get_facecolor(), edgecolor="grey", width=barWidth, label="Single-agent") 
                axes[i].bar(br2, df_ma_col, color=legend_archs[1].get_facecolor(), edgecolor="grey", width=barWidth, label="Multi-agent") 
                axes[i].set_title(beautify_col(it_col))
                axes[i].set_xticks([j + (barWidth/2) for j in br1], br1 + 1)
                axes[i].set_ylim(0, 280)
                axes[i].legend(handles=legend_archs, loc="upper center")
            axes[-1].set_xlabel("Iterations")
            plt.tight_layout()
            plt.show()

        for i in range(len(cols)):
            col = cols[i]
            ticks = bins[i]
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
                c = legend_archs[j].get_facecolor()
                bp["boxes"][j].set_facecolor(c)
            plt.title(beautify_col(col))
            plt.show()
            ## Histogram
            for j in range(len(data)):
                c = legend_archs[j].get_facecolor()
                sns.histplot(data[j], color=c, edgecolor=c, label=archs[j], bins=ticks, kde=True)
            plt.title(f"{beautify_col(col)} (KDE)")
            plt.xlabel("")
            plt.xticks(ticks)
            plt.legend()
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
        #"tfl": ["type", "llm", "file_format"],
        #"tl": ["type", "llm"],
        "tf": ["type", "file_format"],
        "lf": ["llm", "file_format"],
        "t": ["type"],
        "l": ["llm"]
    }

    for name, group in groups.items():
        # group aggregation
        df_group = df_new.groupby(group).agg(
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
            "std_compilation_iteration": "STD_cmpl_iter",
            "avg_testing_iteration": "avg_test_iter", 
            "std_testing_iteration": "STD_test_iter",
            #"avg_validation_iteration": "avg_vald_iter", 
            #"std_validation_iteration": "STD_vald_iter",
            "avg_cyclomatic_complexity": "avg_cyc_cmplx",
            "std_cyclomatic_complexity": "STD_cyc_cmplx",
            "avg_code_coverage": "avg_cod_cov",
            "std_code_coverage": "STD_cod_cov",
            "avg_end_seconds": "avg_end_time", 
            "std_end_seconds": "STD_end_time",
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
