import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
from matplotlib.patches import Patch
from scipy.stats import t, ttest_ind, f_oneway, bootstrap
from pingouin import pairwise_gameshowell



REPLACE_NA = False
SHOW_PLOTS = False
SHOW_BARPLOTS = False

def beautify_col(col: str) -> str:
    return col.capitalize().replace("_", " ")

def cohens_d(x1: list[float], x2: list[float]) -> float:
    n_x1 = len(x1)
    n_x2 = len(x2)

    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)

    sd_x1 = np.std(x1, ddof=1)
    sd_x2 = np.std(x2, ddof=1)

    # pooled standard deviation
    sd_pooled = np.sqrt(
        ((n_x1 - 1) * sd_x1**2 + (n_x2 - 1) * sd_x2**2) / (n_x1 + n_x2 - 2)
    )

    cohens_d = (mean_x1 - mean_x2) / sd_pooled
    return cohens_d

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
    df["execution_time"] = (df["end_time"] - df["start_time"]).dt.total_seconds().astype("float64")

    # drops
    df.drop(columns=[
        # useless
        "best_parser_folder", "testing_rate",
        # not useful anymore
        "start_time", "compilation_time", "testing_time", "validation_time", "end_time"
    ], inplace=True)

    # save
    df.to_csv(benchmarks_dir / f"benchmarks_new.csv", encoding="utf-8", sep=",", na_rep="", float_format="%.4f", index=False)
    
    # calculate correlations
    df_corr = df[[
        "compilation_iteration", 
        "testing_iteration", 
        "cyclomatic_complexity", 
        "code_coverage",
        "execution_time"
    ]].rename(columns={
        "compilation_iteration": "Comp. Iter.", 
        "testing_iteration": "Test. Iter.", 
        "cyclomatic_complexity": "Cyc. Complex.",
        "code_coverage": "Code Cov.",
        "execution_time": "Time"
    }).corr()

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
        br3 = [i + barWidth for i in br2]
        binsCyc = [1, 10, 20, 50, 180]
        #binsCyc = [1, 11, 21, 51, 171]
        #binsCyc = range(1, 172, 10)
        binsCod = np.arange(0, 1.1, 0.1)
        binsExt = np.arange(0, 2281, 120)
        bins = [binsCyc, binsCod, binsExt]
        cols = ["cyclomatic_complexity", "code_coverage", "execution_time"]
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
        ax = sns.heatmap(df_corr, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1, center=0, annot_kws={"size": 14})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.title("Correlation Heatmap")
        plt.show()

        #raise SystemExit
        
        if False:
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
                #sns.histplot(np.log(data[j]), color=c, edgecolor=c, label=archs[j], kde=True)
                sns.histplot(data[j], color=c, edgecolor=c, label=archs[j], bins=ticks, kde=True)
            plt.title(f"{beautify_col(col)} (KDE)")
            plt.xlabel("")
            plt.xticks(ticks)
            plt.legend()
            plt.tight_layout()
            plt.show()

    #raise SystemExit

    models = {
        #"compilation_iteration": {
        #    "model": sm.families.Poisson(),
        #    "model_name": "Poisson",
        #    "link": "log",
        #    "effect_name": "Rate ratio"
        #}, 
        "testing_iteration": {
            "model": sm.families.NegativeBinomial(alpha=0.2),
            "model_name": "Negative Binomial",
            "link": "log",
            "effect_name": "Rate ratio"
        },
        "execution_time": {
            "model": sm.families.Gaussian(link=sm.families.links.Log()),
            "model_name": "Gaussian",
            "link": "log",
            "effect_name": "Mean ratio"
        }, 
        "cyclomatic_complexity": {
            "model": sm.families.Gaussian(link=sm.families.links.Log()),
            "model_name": "Gaussian",
            "link": "log",
            "effect_name": "Mean ratio"
        },
        #"code_coverage": {
        #    "model": sm.families.Gaussian(),
        #    "model_name": "Gaussian",
        #    "link": "identity",
        #    "effect_name": "Mean diff."
        #}
    }

    df_t_count = pd.DataFrame()
    for outcome, obj in models.items():
        model = smf.glm(
            formula=f"{outcome} ~ type + llm + file_format",
            data=df_new,
            family=obj["model"]
        ).fit()
        print(model.summary())

        marginal = model.get_margeff(at='overall')
        print(marginal.summary())

        effect = model.params["type[T.single_agent]"]
        ci = model.conf_int().loc["type[T.single_agent]"]
        if obj["link"] == "log":
            effect = np.exp(effect)
            ci = np.exp(ci)

        d = pd.DataFrame([{
            "Metric": beautify_col(outcome),
            "Model": obj["model_name"],
            "Link": obj["link"],
            "$p$-value": model.pvalues["type[T.single_agent]"],
            #"Effect name": obj["effect_name"],
            "Effect": effect,
            "$CI_l$": ci.values[0],
            "$CI_u$": ci.values[1]
        }])
        df_t_count = pd.concat([df_t_count, d])

    print(df_t_count)
    df_t_count.set_index("Metric", inplace=True)
    with open(benchmarks_dir / f"benchmarks_t_count.tex", "w", encoding="utf-8") as f:
        f.write(df_t_count.to_latex(None, float_format="%.3f"))

    #raise SystemExit

    metrics = [
        "compilation_iteration",
        "testing_iteration",
        "cyclomatic_complexity",
        "code_coverage",
        "execution_time"
    ]
    alpha = 0.05

    for m in metrics:
        print(m)
        x1 = df_new.loc[df_new["type"] == "multi_agent", m].dropna()
        x2 = df_new.loc[df_new["type"] == "single_agent", m].dropna()
        for data in [x1, x2]:
            result = bootstrap(
                (data,),
                statistic = np.mean,
                confidence_level=0.95,
                n_resamples=10000,
                method="BCa",
                random_state=42
            )
            print(result.confidence_interval)

    if False:
        for m in metrics:
            print(m)
            x1 = df_new.loc[df_new["llm"] == "anthropic", m].dropna()
            x2 = df_new.loc[df_new["llm"] == "google", m].dropna()
            x3 = df_new.loc[df_new["llm"] == "openai", m].dropna()
            for data in [x1, x2, x3]:
                result = bootstrap(
                    (data,),
                    statistic = np.mean,
                    confidence_level=0.95,
                    n_resamples=10000,
                    method="BCa",
                    random_state=42
                )
                print(result.confidence_interval)
    
    #raise SystemExit

    # name friendly
    col = "type"
    df_new.loc[df_new[col] == "single_agent", col] = "Single-agent"
    df_new.loc[df_new[col] == "multi_agent", col] = "Multi-agent"
    col = "llm"
    df_new.loc[df_new[col] == "anthropic", col] = "Anthropic"
    df_new.loc[df_new[col] == "google", col] = "Google"
    df_new.loc[df_new[col] == "openai", col] = "OpenAI"
    df_new.rename(columns={
        "type": "Architecture",
        "file_format": "File format",
        "llm": "LLM"
    }, inplace=True)

    df_t_tests = pd.DataFrame()
    #df_l_tests = pd.DataFrame()
    for m in metrics:
        print(m)
        x1 = df_new.loc[df_new["Architecture"] == "Multi-agent", m].dropna()
        x2 = df_new.loc[df_new["Architecture"] == "Single-agent", m].dropna()
        if False:
            # it's the same damn thing
            res = ttest_ind(
                x1,
                x2,
                equal_var=False, # Welch t-test
                nan_policy="omit"
            )
            print(res)
            print(res.confidence_interval())
            print(f"Cohen's d: {cohens_d(x1, x2)}") # this needs for dropna
        res = f_oneway(
            x1,
            x2,
            equal_var=False, # Welch ANOVA
            nan_policy="omit"
        )
        print(res)
        d = pairwise_gameshowell(dv=m, between="Architecture", data=df_new, effsize="cohen")
        d["Metric"] = beautify_col(m)
        d["CI"] = t.ppf(1 - (alpha / 2), df=d["df"]) * d["se"]
        d["CI_l"] = d["diff"] - d["CI"]
        d["CI_u"] = d["diff"] + d["CI"]
        df_t_tests = pd.concat([df_t_tests, d])
        print(d)
        if False:
            print("")
            res = f_oneway(
                df_new.loc[df_new["LLM"] == "Anthropic", m],
                df_new.loc[df_new["LLM"] == "Google", m],
                df_new.loc[df_new["LLM"] == "OpenAI", m],
                equal_var=False, # Welch ANOVA
                nan_policy="omit"
            )
            print(res)
            d = pairwise_gameshowell(dv=m, between="LLM", data=df_new, effsize="cohen")
            d["Metric"] = beautify_col(m)
            d["CI"] = t.ppf(1 - (alpha / 2), df=d["df"]) * d["se"]
            d["CI_l"] = d["diff"] - d["CI"]
            d["CI_u"] = d["diff"] + d["CI"]
            df_l_tests = pd.concat([df_l_tests, d])
            print(d)
        print("\n\n")
    
    df_t_tests.set_index("Metric", inplace=True)
    #df_l_tests.set_index("Metric", inplace=True)
    df_tests_map = {
        "A": "C$_1$", 
        "B": "C$_2$", 
        "CI_l": r"$\Delta CI_l$", 
        "CI_u": r"$\Delta CI_u$",
        "pval": "$p$-value",
        "cohen": "Cohen's $d$"
    }
    df_t_tests = df_t_tests[list(df_tests_map.keys())].rename(columns=df_tests_map)
    #df_l_tests = df_l_tests[list(df_tests_map.keys())].rename(columns=df_tests_map)
    with open(benchmarks_dir / f"benchmarks_t_tests.tex", "w", encoding="utf-8") as f:
        f.write(df_t_tests.to_latex(None, float_format="%.3f"))
    #with open(benchmarks_dir / f"benchmarks_l_tests.tex", "w", encoding="utf-8") as f:
    #    f.write(df_l_tests.to_latex(None, float_format="%.3f"))

    #raise SystemExit

    groups = {
        #"tfl": ["Architecture", "LLM", "File format"],
        #"tl": ["Architecture", "LLM"],
        #"tf": ["Architecture", "File format"],
        #"lf": ["LLM", "File format"],
        "t": ["Architecture"],
        #"l": ["LLM"]
    }

    for name, group in groups.items():
        # group aggregation
        # NB: pandas default ddof for std is equal to 1 (ok for sample std)
        df_group = df_new.groupby(group).agg(
            cnt_all=("n", "count"),

            avg_compilation_iteration=("compilation_iteration", "mean"),
            std_compilation_iteration=("compilation_iteration", "std"),
            med_compilation_iteration=("compilation_iteration", "median"),
            cnt_compilation_iteration=("compilation_iteration", "count"),

            avg_testing_iteration=("testing_iteration", "mean"),
            std_testing_iteration=("testing_iteration", "std"),
            med_testing_iteration=("testing_iteration", "median"),
            cnt_testing_iteration=("testing_iteration", "count"),

            avg_cyclomatic_complexity=("cyclomatic_complexity", "mean"),
            std_cyclomatic_complexity=("cyclomatic_complexity", "std"),
            med_cyclomatic_complexity=("cyclomatic_complexity", "median"),
            cnt_cyclomatic_complexity=("cyclomatic_complexity", "count"),

            avg_code_coverage=("code_coverage", "mean"),
            std_code_coverage=("code_coverage", "std"),
            med_code_coverage=("code_coverage", "median"),
            cnt_code_coverage=("code_coverage", "count"),

            avg_execution_time=("execution_time", "mean"),
            std_execution_time=("execution_time", "std"),
            med_execution_time=("execution_time", "median"),
            cnt_execution_time=("execution_time", "count")
        )
        # group rate calculation
        df_group["compilation_rate"] = df_group["cnt_compilation_iteration"] / df_group["cnt_all"]
        df_group["testing_rate"] = df_group["cnt_testing_iteration"] / df_group["cnt_all"]
        df_group.drop(columns=["cnt_all"], inplace=True)

        for m in metrics:
            # parameters
            means = df_group[f"avg_{m}"]
            stds = df_group[f"std_{m}"]
            ns = df_group[f"cnt_{m}"]
            se = stds / np.sqrt(ns) # standard error
            ci = t.ppf(1 - (alpha / 2), df=ns-1) * se
            # calculations
            df_group[f"lcb_{m}"] = means - ci
            df_group[f"ucb_{m}"] = means + ci

        # reorder columns
        df_group = df_group[[
            "compilation_rate",
            "cnt_compilation_iteration", 
            "avg_compilation_iteration", 
            "std_compilation_iteration",
            "med_compilation_iteration",
            "lcb_compilation_iteration",
            "ucb_compilation_iteration",
            "testing_rate",
            "cnt_testing_iteration", 
            "avg_testing_iteration", 
            "std_testing_iteration",
            "med_testing_iteration",
            "lcb_testing_iteration",
            "ucb_testing_iteration",
            "cnt_cyclomatic_complexity",
            "avg_cyclomatic_complexity",
            "std_cyclomatic_complexity",
            "med_cyclomatic_complexity",
            "lcb_cyclomatic_complexity",
            "ucb_cyclomatic_complexity",
            "cnt_code_coverage",
            "avg_code_coverage",
            "std_code_coverage",
            "med_code_coverage",
            "lcb_code_coverage",
            "ucb_code_coverage",
            "cnt_execution_time", 
            "avg_execution_time", 
            "std_execution_time",
            "med_execution_time",
            "lcb_execution_time",
            "ucb_execution_time"
        ]]

        # print
        #df_group.info()
        #print(df_group)

        # save HTML (more friendly columns names)
        df_html = df_group.rename(columns={
            "compilation_rate": "Compilation rate",
            "cnt_compilation_iteration": "n Compilation iterations",
            "avg_compilation_iteration": "μ Compilation iterations", 
            "std_compilation_iteration": "σ Compilation iterations",
            "med_compilation_iteration": "η Compilation iterations",
            "lcb_compilation_iteration": "LCB Compilation iterations",
            "ucb_compilation_iteration": "UCB Compilation iterations",
            "testing_rate": "Testing rate",
            "cnt_testing_iteration": "n Testing iterations", 
            "avg_testing_iteration": "μ Testing iterations", 
            "std_testing_iteration": "σ Testing iterations",
            "med_testing_iteration": "η Testing iterations",
            "lcb_testing_iteration": "LCB Testing iterations",
            "ucb_testing_iteration": "UCB Testing iterations",
            "cnt_cyclomatic_complexity": "n Cyclomatic complexity",
            "avg_cyclomatic_complexity": "μ Cyclomatic complexity",
            "std_cyclomatic_complexity": "σ Cyclomatic complexity",
            "med_cyclomatic_complexity": "η Cyclomatic complexity",
            "lcb_cyclomatic_complexity": "LCB Cyclomatic complexity",
            "ucb_cyclomatic_complexity": "UCB Cyclomatic complexity",
            "cnt_code_coverage": "n Code coverage",
            "avg_code_coverage": "μ Code coverage",
            "std_code_coverage": "σ Code coverage",
            "med_code_coverage": "η Code coverage",
            "lcb_code_coverage": "LCB Code coverage",
            "ucb_code_coverage": "UCB Code coverage",
            "cnt_execution_time": "n Execution time", 
            "avg_execution_time": "μ Execution time", 
            "std_execution_time": "σ Execution time",
            "med_execution_time": "η Execution time",
            "lcb_execution_time": "LCB Execution time",
            "ucb_execution_time": "UCB Execution time"
        })

        # overall
        if False and name in ["tfl", "tf", "lf"]:
            if name == "tfl":
                ind = ["LLM", "Architecture"]
            elif name == "tf":
                ind = ["Architecture"]
            else:
                ind = ["LLM"]
            html = "<h1>Benchmarks</h1>"
            for col in [
                "Compilation rate", 
                "Testing rate", 
                "μ Compilation iterations",
                "μ Testing iterations",
                "μ Cyclomatic complexity",
                "μ Code coverage",
                "μ Execution time"
            ]:
                if "rate" in col:
                    map = "RdYlGn"
                    min = 0
                    max = 1
                elif "iterations" in col:
                    map = "RdYlGn_r"
                    min = 1
                    max = 15
                elif "Cyclomatic" in col:
                    map = "RdYlGn_r"
                    min = 1
                    max = 100
                elif "coverage" in col:
                    map = "RdYlGn"
                    min = 0
                    max = 1
                else:
                    map = "RdYlGn_r"
                    min = 1
                    max = 1000

                html += f"<br><h2>{col}</h2>" + df_html[[col]].reset_index().pivot_table(
                    index=ind,
                    columns="File format",
                    values=col
                ).style.background_gradient(
                    cmap=map, 
                    vmin=min, 
                    vmax=max
                ).format("{:.3f}").to_html(
                    encoding="utf-8", 
                    border=False
                )

            with open(benchmarks_dir / f"benchmarks_{name}.html", "w", encoding="utf-8") as f:
                f.write(html)

        # only rates
        df_html_rates = df_html[[
            "Compilation rate",
            "Testing rate"
        ]]
        df_html_rates.style.background_gradient(
            cmap ="RdYlGn", 
            vmin=0, 
            vmax=1
        ).format("{:.3f}").to_html(
            benchmarks_dir / f"benchmarks_{name}_1.html", 
            encoding="utf-8", 
            border=False
        )
        df_html_rates.to_latex(
            benchmarks_dir / f"benchmarks_{name}_rates.tex", 
            encoding="utf-8", 
            float_format="%.3f"
        )
        
        # else (and reordered)
        df_html = df_html[[
            "n Compilation iterations", 
            "η Compilation iterations",
            "μ Compilation iterations", 
            "σ Compilation iterations",
            "LCB Compilation iterations",
            "UCB Compilation iterations",
            "n Testing iterations", 
            "η Testing iterations",
            "μ Testing iterations", 
            "σ Testing iterations",
            "LCB Testing iterations",
            "UCB Testing iterations",
            "n Execution time", 
            "η Execution time",
            "μ Execution time", 
            "σ Execution time",
            "LCB Execution time",
            "UCB Execution time",
            "n Cyclomatic complexity",
            "η Cyclomatic complexity",
            "μ Cyclomatic complexity",
            "σ Cyclomatic complexity",
            "LCB Cyclomatic complexity",
            "UCB Cyclomatic complexity",
            "n Code coverage",
            "η Code coverage",
            "μ Code coverage",
            "σ Code coverage",
            "LCB Code coverage",
            "UCB Code coverage"
        ]]
        
        metrics_out = [
            "Compilation iterations",
            "Testing iterations",
            "Execution time",
            "Cyclomatic complexity",
            "Code coverage"
        ]
        my_list = []
        for mo in metrics_out:
            # NB: same order
            my_list.append((mo, "n"))
            my_list.append((mo, "η"))
            my_list.append((mo, "μ"))
            my_list.append((mo, "σ"))
            my_list.append((mo, "LCB"))
            my_list.append((mo, "UCB"))
        df_html.columns = pd.MultiIndex.from_tuples(my_list)

        # only iterations
        df_html[[
            "Compilation iterations",
            "Testing iterations",
            "Execution time"
        ]].to_html(
            benchmarks_dir / f"benchmarks_{name}_2.html", 
            encoding="utf-8", 
            float_format="%.3f", 
            border=False
        )
        # only vulnerability
        df_html[[
            "Cyclomatic complexity",
            "Code coverage"
        ]].to_html(
            benchmarks_dir / f"benchmarks_{name}_3.html", 
            encoding="utf-8", 
            float_format="%.3f", 
            border=False
        )

        # for latex (convert population symbols to sample symbols)
        df_html.rename(columns={
            "μ": r"$\bar{x}$", 
            "σ": "$s$", 
            "η": r"$\tilde{x}$", 
            "LCB": "$CI_l$", 
            "UCB": "$CI_u$"
        }, level=1, inplace=True)
        df_latex = pd.DataFrame()
        for col in [
            "Compilation iterations",
            "Testing iterations",
            "Execution time",
            "Cyclomatic complexity",
            "Code coverage"
        ]:
            df_latex_sub = df_html[col].copy()
            df_latex_sub["Metric"] = col
            df_latex = pd.concat([df_latex, df_latex_sub])
        df_latex.reset_index(inplace=True)
        df_latex.set_index(["Metric"] + group, inplace=True)
        with open(benchmarks_dir / f"benchmarks_{name}.tex", "w", encoding="utf-8") as f:
            f.write(df_latex.to_latex(None, float_format="%.3f", multirow=False))
