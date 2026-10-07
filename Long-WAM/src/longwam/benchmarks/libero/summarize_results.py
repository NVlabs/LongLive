# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: experiments/libero/summarize_results.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import os
import json
import argparse
from collections import defaultdict
import pandas as pd
import math


def format_time(seconds):
    """Format seconds as a human-readable duration string.

    - Below 1 minute: SS
    - Below 1 hour: MMSS
    - 1 hour or longer: HHMMSS
    """
    seconds = round(seconds)  # Round to integer seconds

    if seconds < 60:
        return f"{seconds:02d}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        remaining_seconds = remaining % 60
        return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"


def summarize_results(output_dir):
    """Summarize all evaluation results.

    Args:
        output_dir: Root directory containing result files.
    """
    # Store statistics for each suite
    suite_stats = defaultdict(
        lambda: {
            "total_tasks": 0,
            "total_trials": 0,
            "total_successes": 0,
            "total_time": 0,
            "max_time": 0,
            "psnr_sum": 0.0,
            "psnr_count": 0,
        }
    )

    # Store detailed per-task results
    task_results = {}
    has_psnr_metric = False

    # Iterate over all suite directories
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        suite_dir = os.path.join(output_dir, suite)
        if not os.path.exists(suite_dir):
            continue

        # Read all result files
        for filename in os.listdir(suite_dir):
            if not filename.startswith("gpu") or not filename.endswith("_results.json"):
                continue

            with open(os.path.join(suite_dir, filename), "r") as f:
                result = json.load(f)

            # Extract task ID from the filename
            parts = filename.split("_")
            task_id = int(parts[1].replace("task", ""))

            # Create the task identifier (suite_taskid)
            task_key = f"{suite}_{task_id}"

            stats = suite_stats[suite]
            stats["total_tasks"] += 1
            stats["total_trials"] += result["total_episodes"]
            stats["total_successes"] += result["successes"]
            stats["total_time"] += result["duration"]
            stats["max_time"] = max(stats["max_time"], result["duration"])
            if "future_video_psnr_mean" in result:
                has_psnr_metric = True
                if result["future_video_psnr_mean"] is not None:
                    stats["psnr_sum"] += float(result["future_video_psnr_mean"])
                    stats["psnr_count"] += 1

            # Store detailed task results
            task_result = {
                "success_rate": result["successes"] / result["total_episodes"] * 100,
                "duration": result["duration"],
                "total_episodes": result["total_episodes"],
                "successes": result["successes"],
                "task_description": result["task_description"]
                if "task_description" in result
                else "",
            }
            if "future_video_psnr_mean" in result:
                task_result["future_video_psnr_mean"] = (
                    float(result["future_video_psnr_mean"])
                    if result["future_video_psnr_mean"] is not None
                    else None
                )
            task_results[task_key] = task_result

    # Print summary results
    print("\n=== Evaluation Results Summary ===")
    print("\nStatistics for each task suite:")

    total_success_rate = 0
    total_time = 0
    total_suites = 0
    overall_psnr_sum = 0.0
    overall_psnr_count = 0

    # Prepare DataFrame rows
    df_data = {"Task Suite": [], "Success Rate (%)": [], "Average Time (s)": [], "Max Time (s)": []}
    if has_psnr_metric:
        df_data["Average Future PSNR (dB)"] = []

    for suite, stats in suite_stats.items():
        if stats["total_trials"] > 0:
            success_rate = stats["total_successes"] / stats["total_trials"] * 100
            avg_time = stats["total_time"] / stats["total_tasks"]
            max_time = stats["max_time"]
            suite_avg_psnr = None
            if has_psnr_metric:
                suite_avg_psnr = (
                    stats["psnr_sum"] / stats["psnr_count"] if stats["psnr_count"] > 0 else None
                )

            print(f"\n{suite}:")
            print(f"- Tasks completed: {stats['total_tasks']}")
            print(f"- Total attempts: {stats['total_trials']}")
            print(f"- Successful attempts: {stats['total_successes']}")
            print(f"- Success rate: {success_rate:.2f}%")
            print(f"- Total time: {format_time(stats['total_time'])}")
            print(f"- Average time per task: {format_time(avg_time)}")
            print(f"- Longest task time: {format_time(max_time)}")
            if has_psnr_metric:
                if suite_avg_psnr is not None:
                    print(f"- Average future-video PSNR: {suite_avg_psnr:.4f} dB")
                else:
                    print("- Average future-video PSNR: N/A")

            # Append to DataFrame rows
            df_data["Task Suite"].append(suite)
            df_data["Success Rate (%)"].append(f"{success_rate:.2f}")
            df_data["Average Time (s)"].append(f"{avg_time:.2f}")
            df_data["Max Time (s)"].append(f"{max_time:.2f}")
            if has_psnr_metric:
                df_data["Average Future PSNR (dB)"].append(
                    f"{suite_avg_psnr:.4f}" if suite_avg_psnr is not None else "N/A"
                )

            total_success_rate += success_rate
            total_time += stats["total_time"]
            total_suites += 1
            if has_psnr_metric:
                overall_psnr_sum += stats["psnr_sum"]
                overall_psnr_count += stats["psnr_count"]

    if total_suites > 0:
        print("\nOverall statistics:")
        avg_success_rate = total_success_rate / total_suites
        avg_task_time = total_time / sum(s["total_tasks"] for s in suite_stats.values())
        max_task_time = max(s["max_time"] for s in suite_stats.values())
        overall_avg_psnr = None
        if has_psnr_metric:
            overall_avg_psnr = (
                overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
            )

        print(f"- Average success rate: {avg_success_rate:.2f}%")
        print(f"- Total time: {format_time(total_time)}")
        print(f"- Average time per task: {format_time(avg_task_time)}")
        print(f"- Longest task time: {format_time(max_task_time)}")
        if has_psnr_metric:
            if overall_avg_psnr is not None:
                print(f"- Average future-video PSNR: {overall_avg_psnr:.4f} dB")
            else:
                print("- Average future-video PSNR: N/A")

        # Add an overall summary row
        df_data["Task Suite"].append("Overall")
        df_data["Success Rate (%)"].append(f"{avg_success_rate:.2f}")
        df_data["Average Time (s)"].append(f"{avg_task_time:.2f}")
        df_data["Max Time (s)"].append(f"{max_task_time:.2f}")
        if has_psnr_metric:
            df_data["Average Future PSNR (dB)"].append(
                f"{overall_avg_psnr:.4f}" if overall_avg_psnr is not None else "N/A"
            )

    # Create and save the DataFrame
    df = pd.DataFrame(df_data)

    # Use the last checkpoint path component as the title
    ckpt_path = os.environ.get("CKPT", "")
    title = os.path.basename(ckpt_path) if ckpt_path else "Results"

    # Transpose the DataFrame and use Task Suite as column names
    df = df.set_index("Task Suite").T

    # Add a title line to the CSV file
    with open(os.path.join(output_dir, "summary.csv"), "w") as f:
        f.write(f"{title}\n")  # Write the title
        df.to_csv(f)

    # Create the per-task success-rate CSV
    task_success_data = {"Task": [], "Description": [], "Success Rate (%)": []}
    if has_psnr_metric:
        task_success_data["Future Video PSNR (dB)"] = []

    # Group tasks by suite
    suite_tasks = defaultdict(list)
    for task in task_results:
        suite = task.split("_")[0] + "_" + task.split("_")[1]
        suite_tasks[suite].append(task)

    # Sort tasks within each suite
    for suite in suite_tasks:
        suite_tasks[suite].sort(key=lambda x: int(x.split("_")[-1]))

    # Fill per-task success-rate rows
    for suite in sorted(suite_tasks.keys()):
        for task in suite_tasks[suite]:
            result = task_results[task]
            task_success_data["Task"].append(task)
            task_success_data["Description"].append(
                result["task_description"] if "task_description" in result else ""
            )
            task_success_data["Success Rate (%)"].append(f"{result['success_rate']:.2f}")
            if has_psnr_metric:
                psnr = (
                    result["future_video_psnr_mean"] if "future_video_psnr_mean" in result else None
                )
                task_success_data["Future Video PSNR (dB)"].append(
                    f"{psnr:.4f}" if psnr is not None else "N/A"
                )

    suite_stats_output = {}
    for suite, stats in suite_stats.items():
        suite_stats_output[suite] = {
            "total_tasks": stats["total_tasks"],
            "total_trials": stats["total_trials"],
            "total_successes": stats["total_successes"],
            "total_time": stats["total_time"],
            "max_time": stats["max_time"],
        }
        if has_psnr_metric:
            suite_stats_output[suite]["average_future_video_psnr"] = (
                stats["psnr_sum"] / stats["psnr_count"] if stats["psnr_count"] > 0 else None
            )

    # Create and save the task success-rate DataFrame
    task_success_df = pd.DataFrame(task_success_data)
    task_success_df.to_csv(os.path.join(output_dir, "task_success_rates.csv"), index=False)

    # Save the detailed JSON summary
    summary_file = os.path.join(output_dir, "summary.json")
    overall_stats = {
        "average_success_rate": total_success_rate / total_suites if total_suites > 0 else 0,
        "total_time": total_time,
        "average_task_time": total_time / sum(s["total_tasks"] for s in suite_stats.values())
        if suite_stats
        else 0,
    }
    if has_psnr_metric:
        overall_stats["average_future_video_psnr"] = (
            overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
        )

    with open(summary_file, "w") as f:
        json.dump(
            {
                "run_id": os.path.basename(output_dir),
                "ckpt": os.environ.get("CKPT", ""),  # Checkpoint path
                "config": os.environ.get("CONFIG", ""),  # Config path
                "suite_stats": suite_stats_output,
                "task_results": task_results,
                "overall": overall_stats,
            },
            f,
            indent=4,
        )

    print(f"\n=== Run Information ===")
    print(f"Run ID: {os.path.basename(output_dir)}")
    print(f"Results directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Summary CSV: {os.path.join(output_dir, 'summary.csv')}")
    print(f"Task success rates CSV: {os.path.join(output_dir, 'task_success_rates.csv')}")

    # Print the task success-rate table
    print("\n=== Task Success Rates ===")
    print(task_success_df.to_string(index=False))

    # Print the transposed summary table
    print("\n=== Results Table ===")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Root directory containing evaluation results"
    )
    args = parser.parse_args()

    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
