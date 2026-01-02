import re
from csv import DictReader, DictWriter
from pathlib import Path
from lizard import analyze_file
from utils.general import analyze_c_code, get_c_parser_path



if __name__ == "__main__":
    # parameters
    benchmarks_dir = Path("benchmark")
    benchmarks = []

    # read
    fr = open(benchmarks_dir / "benchmark.csv", encoding="utf-8", newline="")
    reader = DictReader(fr)

    # edit
    coc_pattern = "Lines executed:"
    for row in reader:
        parser_path = Path(row["best_parser_folder"])
        c_parser_path_str = str(get_c_parser_path(parser_path))
        # Cyclomatic Complexity
        cyc_analyzer = analyze_file(c_parser_path_str)
        cyc_list = [ cyc_f.cyclomatic_complexity for cyc_f in cyc_analyzer.function_list ]
        cyc_list = [ cyc_item for cyc_item in cyc_list if cyc_item is not None ]
        if cyc_list:
            row["cyclomatic_complexity"] = max(cyc_list)
        # Code Coverage
        if row["testing_time"]:
            result = analyze_c_code(parser_path, row["file_format"])
            lines = result.splitlines()
            lines = [line for line in lines if line.startswith(coc_pattern)]
            if lines:
                matches = re.search(rf"{coc_pattern}([\d.]+)% of (\d+)", lines[-1])
                if matches:
                    row["code_coverage"] = float(matches.group(1))
        benchmarks.append(row)
        
    # close
    fr.close()

    #print(benchmarks)

    # write
    fw = open(benchmarks_dir / "benchmark_edit.csv", "w", encoding="utf-8", newline="")
    writer = DictWriter(fw, fieldnames=benchmarks[0].keys())
    writer.writeheader()
    writer.writerows(benchmarks)

    # close
    fw.close()