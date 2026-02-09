import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch
from scipy import stats


REPLACE_NA = False
SHOW_PLOTS = False
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
    df_corr = df[[
        "compilation_iteration", 
        "testing_iteration", 
        "cyclomatic_complexity", 
        "code_coverage",
        "end_seconds"
    ]].rename(columns={
        "compilation_iteration": "Comp. Iter.", 
        "testing_iteration": "Test. Iter.", 
        "cyclomatic_complexity": "Cyc. Complex.",
        "code_coverage": "Code Cov.",
        "end_seconds": "Time"
    }).corr()
    df_corr.to_csv(benchmarks_dir / f"benchmarks_corr.csv", encoding="utf-8", sep=",", na_rep="", float_format="%.4f")

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
        median_style = {"color": "red", "linewidth": 1.5}

        # Set global window size
        plt.rcParams.update({
            "figure.figsize": (11, 7),
            "figure.dpi": 100
        })

        # Correlation heatmap plot
        mask = np.triu(df_corr)
        np.fill_diagonal(mask, 0)
        ax = sns.heatmap(df_corr, mask=mask, annot=True, cmap="RdYlGn", vmin=-1, vmax=1, center=0, annot_kws={"size": 12})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.title("Correlation Heatmap")
        plt.show()

        #raise SystemExit
        
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
        
        #raise SystemExit

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

            avg_compilation_iteration=("compilation_iteration", "mean"),
            std_compilation_iteration=("compilation_iteration", "std"),
            cnt_compilation_iteration=("compilation_iteration", "count"),

            avg_testing_iteration=("testing_iteration", "mean"),
            std_testing_iteration=("testing_iteration", "std"),
            cnt_testing_iteration=("testing_iteration", "count"),

            avg_cyclomatic_complexity=("cyclomatic_complexity", "mean"),
            std_cyclomatic_complexity=("cyclomatic_complexity", "std"),
            cnt_cyclomatic_complexity=("cyclomatic_complexity", "count"),

            avg_code_coverage=("code_coverage", "mean"),
            std_code_coverage=("code_coverage", "std"),
            cnt_code_coverage=("code_coverage", "count"),

            avg_end_seconds=("end_seconds", "mean"),
            std_end_seconds=("end_seconds", "std"),
            cnt_end_seconds=("end_seconds", "count")
        )
        # group rate calculation
        df_group["compilation_rate"] = df_group["cnt_compilation_iteration"] / df_group["cnt_all"]
        df_group["testing_rate"] = df_group["cnt_testing_iteration"] / df_group["cnt_all"]
        #df_group["testing_rate_abs"] = df_group["cnt_testing_iteration"] / df_group["cnt_all"]
        #df_group["testing_rate_rel"] = df_group["cnt_testing_iteration"] / df_group["cnt_compilation_iteration"]
        df_group.drop(columns=["cnt_all"], inplace=True)

        metrics = [
            "compilation_iteration",
            "testing_iteration",
            "cyclomatic_complexity",
            "code_coverage",
            "end_seconds"
        ]
        alpha = 0.05

        for m in metrics:
            # parameters
            means = df_group[f"avg_{m}"]
            stds = df_group[f"std_{m}"]
            ns = df_group[f"cnt_{m}"]
            se = stds / np.sqrt(ns) # standard error
            ci = stats.t.ppf(1 - (alpha / 2), df=ns-1) * se
            # calculations
            df_group[f"snr_{m}"] = means / stds
            df_group[f"lcb_{m}"] = means - ci
            df_group[f"ucb_{m}"] = means + ci
            # cleaning
            df_group.drop(columns=[f"cnt_{m}"], inplace=True)

        # reorder columns
        df_group = df_group[[
            "compilation_rate",
            "avg_compilation_iteration", 
            "std_compilation_iteration",
            "snr_compilation_iteration",
            "lcb_compilation_iteration",
            "ucb_compilation_iteration",
            "testing_rate",
            "avg_testing_iteration", 
            "std_testing_iteration",
            "snr_testing_iteration",
            "lcb_testing_iteration",
            "ucb_testing_iteration",
            "avg_cyclomatic_complexity",
            "std_cyclomatic_complexity",
            "snr_cyclomatic_complexity",
            "lcb_cyclomatic_complexity",
            "ucb_cyclomatic_complexity",
            "avg_code_coverage",
            "std_code_coverage",
            "snr_code_coverage",
            "lcb_code_coverage",
            "ucb_code_coverage",
            "avg_end_seconds", 
            "std_end_seconds",
            "snr_end_seconds",
            "lcb_end_seconds",
            "ucb_end_seconds"
        ]]

        # print
        df_group.info()
        print(df_group)

        # save CSV (more compact columns names)
        df_csv = df_group.rename(columns={
            "compilation_rate": "cmpl_rate",
            "avg_compilation_iteration": "AVG_cmpl_iter", 
            "std_compilation_iteration": "STD_cmpl_iter",
            "snr_compilation_iteration": "SNR_cmpl_iter",
            "lcb_compilation_iteration": "LCB_cmpl_iter",
            "ucb_compilation_iteration": "UCB_cmpl_iter",
            "testing_rate": "test_rate",
            "avg_testing_iteration": "AVG_test_iter", 
            "std_testing_iteration": "STD_test_iter",
            "snr_testing_iteration": "SNR_test_iter",
            "lcb_testing_iteration": "LCB_test_iter",
            "ucb_testing_iteration": "UCB_test_iter",
            "avg_cyclomatic_complexity": "AVG_cyc_cmplx",
            "std_cyclomatic_complexity": "STD_cyc_cmplx",
            "snr_cyclomatic_complexity": "SNR_cyc_cmplx",
            "lcb_cyclomatic_complexity": "LCB_cyc_cmplx",
            "ucb_cyclomatic_complexity": "UCB_cyc_cmplx",
            "avg_code_coverage": "AVG_cod_cov",
            "std_code_coverage": "STD_cod_cov",
            "snr_code_coverage": "SNR_cod_cov",
            "lcb_code_coverage": "LCB_cod_cov",
            "ucb_code_coverage": "UCB_cod_cov",
            "avg_end_seconds": "AVG_end_time", 
            "std_end_seconds": "STD_end_time",
            "snr_end_seconds": "SNR_end_time",
            "lcb_end_seconds": "LCB_end_time",
            "ucb_end_seconds": "UCB_end_time"
        })
        df_csv.to_csv(benchmarks_dir / f"benchmarks_{name}.csv", encoding="utf-8", sep=",", na_rep="", float_format="%.4f")

        # save HTML (more friendly columns names)
        df_html = df_group.rename(columns={
            "compilation_rate": "Compilation rate",
            "avg_compilation_iteration": "μ Compilation iterations", 
            "std_compilation_iteration": "σ Compilation iterations",
            "snr_compilation_iteration": "SNR Compilation iterations",
            "lcb_compilation_iteration": "LCB Compilation iterations",
            "ucb_compilation_iteration": "UCB Compilation iterations",
            "testing_rate": "Testing rate",
            "avg_testing_iteration": "μ Testing iterations", 
            "std_testing_iteration": "σ Testing iterations",
            "snr_testing_iteration": "SNR Testing iterations",
            "lcb_testing_iteration": "LCB Testing iterations",
            "ucb_testing_iteration": "UCB Testing iterations",
            "avg_cyclomatic_complexity": "μ Cyclomatic complexity",
            "std_cyclomatic_complexity": "σ Cyclomatic complexity",
            "snr_cyclomatic_complexity": "SNR Cyclomatic complexity",
            "lcb_cyclomatic_complexity": "LCB Cyclomatic complexity",
            "ucb_cyclomatic_complexity": "UCB Cyclomatic complexity",
            "avg_code_coverage": "μ Code coverage",
            "std_code_coverage": "σ Code coverage",
            "snr_code_coverage": "SNR Code coverage",
            "lcb_code_coverage": "LCB Code coverage",
            "ucb_code_coverage": "UCB Code coverage",
            "avg_end_seconds": "μ End time (s)", 
            "std_end_seconds": "σ End time (s)",
            "snr_end_seconds": "SNR End time (s)",
            "lcb_end_seconds": "LCB End time (s)",
            "ucb_end_seconds": "UCB End time (s)"
        })
        #df_html.to_html(benchmarks_dir / f"benchmarks_{name}.html", encoding="utf-8", na_rep="", float_format="%.3f", border=False)
        df_html[[
            "Compilation rate",
            "Testing rate"
        ]].to_html(benchmarks_dir / f"benchmarks_{name}_1.html", encoding="utf-8", na_rep="", float_format="%.3f", border=False)
        df_html[[
            "μ Compilation iterations", 
            "σ Compilation iterations",
            "SNR Compilation iterations",
            "LCB Compilation iterations",
            "UCB Compilation iterations",
            "μ Testing iterations", 
            "σ Testing iterations",
            "SNR Testing iterations",
            "LCB Testing iterations",
            "UCB Testing iterations"
        ]].to_html(benchmarks_dir / f"benchmarks_{name}_2.html", encoding="utf-8", na_rep="", float_format="%.3f", border=False)
        df_html[[
            "μ Cyclomatic complexity",
            "σ Cyclomatic complexity",
            "SNR Cyclomatic complexity",
            "LCB Cyclomatic complexity",
            "UCB Cyclomatic complexity",
            "μ Code coverage",
            "σ Code coverage",
            "SNR Code coverage",
            "LCB Code coverage",
            "UCB Code coverage",
            "μ End time (s)", 
            "σ End time (s)",
            "SNR End time (s)",
            "LCB End time (s)",
            "UCB End time (s)"
        ]].to_html(benchmarks_dir / f"benchmarks_{name}_3.html", encoding="utf-8", na_rep="", float_format="%.3f", border=False)
