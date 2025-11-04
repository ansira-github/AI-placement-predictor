# agent.py

import re
import pandas as pd
import matplotlib.pyplot as plt
from llama_cpp import Llama
from backend.tools import predict_placement, get_placement_advice, analyze_improvement_scenarios

# -----------------------------
# Load LLaMA model safely
# -----------------------------
MODEL_PATH = r"C:\Users\anura\Desktop\ai_agent_project\models\llama-2-7b.Q4_K_S.gguf"
try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=8)
except Exception as e:
    print(f"⚠️ Warning: Failed to load LLaMA model: {e}")
    llm = None


def create_agent(user_input: str, df: pd.DataFrame = None):
    """Main agent function. Handles placement, dataset Q&A, plots, or fallback chat."""

    # -------------------------
    # Placement prediction flow
    # -------------------------
    if "placement" in user_input.lower() and ("cgpa" in user_input.lower() or "iq" in user_input.lower()):
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", user_input)
            if len(numbers) >= 2:
                cgpa = float(numbers[0])
                iq = float(numbers[1])

                result, prob, influence = predict_placement(cgpa, iq)
                return {
                    "type": "placement",
                    "cgpa": cgpa,
                    "iq": iq,
                    "placed": bool(result),
                    "confidence": float(prob),
                    "influence": influence,
                    "followups": [
                        f"What if my CGPA was {cgpa + 0.5}?",
                        f"What if my IQ was {iq + 10}?",
                        "Which matters more: CGPA or IQ?",
                        "How can I improve my chances of placement?",
                        "What is the number of students placed and not placed?",
                        "What is the placement percentage?",
                        "Show me a scatter plot of CGPA vs IQ",
                        "Show me average CGPA comparison plot",
                        "Show me average IQ comparison plot",
                        "Summarize the placement dataset"
                    ]
                }
            else:
                return {
                    "type": "placement",
                    "error": "Please provide both CGPA and IQ for placement prediction.",
                }
        except Exception as e:
            return {"type": "placement", "error": f"Error during placement prediction: {e}"}

    # -------------------------
    # Improvement advice
    # -------------------------
    if "improve" in user_input.lower() or "advice" in user_input.lower():
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", user_input)
            if len(numbers) >= 2:
                cgpa = float(numbers[0])
                iq = float(numbers[1])
                result, _, _ = predict_placement(cgpa, iq)
                advice = get_placement_advice(cgpa, iq, result)
                scenarios = analyze_improvement_scenarios(cgpa, iq)
                return {
                    "type": "advice",
                    "response": advice,
                    "scenarios": scenarios,
                    "followups": [
                        "What are the placement trends?",
                        "Show me improvement scenarios"
                    ]
                }
        except Exception as e:
            return {"type": "advice", "response": f"⚠️ Error generating advice: {e}"}

    # -------------------------
    # Which matters more: CGPA vs IQ
    # -------------------------
    if "which matters more" in user_input.lower() or "more important" in user_input.lower():
        if df is not None and not df.empty:
            try:
                placed_df = df[df["Placed"] == "✅ Yes"]
                not_placed_df = df[df["Placed"] == "❌ No"]

                cgpa_diff = placed_df["cgpa"].mean() - not_placed_df["cgpa"].mean()
                iq_diff = placed_df["iq"].mean() - not_placed_df["iq"].mean()

                more_important = "CGPA" if abs(cgpa_diff) > abs(iq_diff) else "IQ"

                response = f"📊 **Feature Importance Analysis:**\n\n"
                response += f"• CGPA difference (Placed vs Not Placed): {cgpa_diff:.2f}\n"
                response += f"• IQ difference (Placed vs Not Placed): {iq_diff:.2f}\n\n"
                response += f"✅ **{more_important}** appears to have a stronger correlation with placement!"

                return {
                    "type": "dataframe",
                    "response": response,
                    "followups": [
                        "Show me average CGPA comparison plot",
                        "Show me average IQ comparison plot"
                    ]
                }
            except Exception as e:
                return {"type": "dataframe", "response": f"⚠️ Error analyzing data: {e}"}

    # -------------------------
    # Dataframe insights + plots
    # -------------------------
    if df is not None and not df.empty:
        lower_q = user_input.lower()

        try:
            # Dataset summary
            if "summarize" in lower_q or "summary" in lower_q:
                placed = (df["Placed"] == "✅ Yes").sum()
                not_placed = (df["Placed"] == "❌ No").sum()
                avg_cgpa = df["cgpa"].mean()
                avg_iq = df["iq"].mean()
                placement_rate = (placed / len(df)) * 100

                return {
                    "type": "dataframe",
                    "response": f"📊 **Dataset Summary:**\n\n"
                                f"• Total students: {len(df)}\n"
                                f"• ✅ Placed: {placed} ({placement_rate:.1f}%)\n"
                                f"• ❌ Not Placed: {not_placed} ({100 - placement_rate:.1f}%)\n"
                                f"• Average CGPA: {avg_cgpa:.2f}\n"
                                f"• Average IQ: {avg_iq:.2f}",
                    "followups": [
                        "What is the placement percentage?",
                        "Show me a bar chart of placement counts",
                        "Show me average CGPA comparison plot",
                        "Show me average IQ comparison plot",
                        "Which matters more: CGPA or IQ?"
                    ]
                }

            # Count stats
            if "number of students" in lower_q or "count" in lower_q:
                placed = (df["Placed"] == "✅ Yes").sum()
                not_placed = (df["Placed"] == "❌ No").sum()
                return {
                    "type": "dataframe",
                    "response": f"✅ Placed: {placed}, ❌ Not Placed: {not_placed}",
                    "followups": [
                        "What is the placement percentage?",
                        "Show me a bar chart of placement counts",
                        "Summarize the placement dataset"
                    ]
                }

            # Percentage
            elif "percentage" in lower_q:
                placed = (df["Placed"] == "✅ Yes").sum()
                total = len(df)
                percent = (placed / total) * 100 if total > 0 else 0
                return {
                    "type": "dataframe",
                    "response": f"📊 Placement Percentage: {percent:.2f}%",
                    "followups": [
                        "Show me a pie chart of placement percentage",
                        "Summarize the placement dataset",
                        "Show me average IQ of placed vs not placed"
                    ]
                }

            # Avg CGPA
            elif "average cgpa" in lower_q and "plot" not in lower_q:
                placed_avg = df[df["Placed"] == "✅ Yes"]["cgpa"].mean()
                not_avg = df[df["Placed"] == "❌ No"]["cgpa"].mean()
                return {
                    "type": "dataframe",
                    "response": f"🎓 Avg CGPA - Placed: {placed_avg:.2f}, Not Placed: {not_avg:.2f}",
                    "followups": ["Show me average CGPA comparison plot"]
                }

            # Avg IQ
            elif "average iq" in lower_q and "plot" not in lower_q:
                placed_avg = df[df["Placed"] == "✅ Yes"]["iq"].mean()
                not_avg = df[df["Placed"] == "❌ No"]["iq"].mean()
                return {
                    "type": "dataframe",
                    "response": f"🧠 Avg IQ - Placed: {placed_avg:.2f}, Not Placed: {not_avg:.2f}",
                    "followups": ["Show me average IQ comparison plot"]
                }

            # Scatter plot CGPA vs IQ
            elif "scatter" in lower_q and "cgpa" in lower_q and "iq" in lower_q:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = df["Placed"].map({"✅ Yes": "#10b981", "❌ No": "#ef4444"})
                ax.scatter(df["cgpa"], df["iq"], c=colors, s=100, alpha=0.6,
                           edgecolors='black', linewidth=0.5)
                ax.set_xlabel("CGPA", fontsize=12, fontweight='bold')
                ax.set_ylabel("IQ", fontsize=12, fontweight='bold')
                ax.set_title("Placement Prediction: CGPA vs IQ", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#10b981', label='✅ Placed'),
                    Patch(facecolor='#ef4444', label='❌ Not Placed')
                ]
                ax.legend(handles=legend_elements, loc='best')

                plt.tight_layout()
                return {
                    "type": "plot",
                    "caption": "📊 Scatter plot of CGPA vs IQ",
                    "figure": fig,
                    "followups": ["Show me average CGPA comparison plot", "Which matters more: CGPA or IQ?"]
                }

            # Bar chart placement counts
            elif "bar chart" in lower_q and "placement" in lower_q:
                fig, ax = plt.subplots(figsize=(8, 6))
                counts = df["Placed"].value_counts()
                counts.plot(kind="bar", ax=ax, color=["#10b981", "#ef4444"])
                ax.set_title("Placement Counts", fontsize=14, fontweight='bold')
                ax.set_ylabel("Number of Students", fontsize=12)
                ax.set_xlabel("Placement Status", fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                return {
                    "type": "plot",
                    "caption": "📊 Bar chart of students placed vs not placed",
                    "figure": fig,
                }

            # Pie chart placement percentage
            elif "pie chart" in lower_q and "placement" in lower_q:
                fig, ax = plt.subplots(figsize=(8, 8))
                df["Placed"].value_counts().plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    colors=["#10b981", "#ef4444"],
                    ax=ax,
                    startangle=90
                )
                ax.set_ylabel("")
                ax.set_title("Placement Distribution", fontsize=14, fontweight='bold')
                plt.tight_layout()
                return {
                    "type": "plot",
                    "caption": "📊 Pie chart of placement percentage",
                    "figure": fig,
                }

            # Avg CGPA comparison plot
            elif "average cgpa" in lower_q and "plot" in lower_q:
                fig, ax = plt.subplots(figsize=(8, 6))
                means = df.groupby("Placed")["cgpa"].mean()
                means.plot(kind="bar", ax=ax, color=["#ef4444", "#10b981"])
                ax.set_title("Average CGPA: Placed vs Not Placed", fontsize=14, fontweight='bold')
                ax.set_ylabel("Average CGPA", fontsize=12)
                ax.set_xlabel("Placement Status", fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                return {
                    "type": "plot",
                    "caption": "🎓 Avg CGPA comparison plot",
                    "figure": fig,
                }

            # Avg IQ comparison plot
            elif "average iq" in lower_q and "plot" in lower_q:
                fig, ax = plt.subplots(figsize=(8, 6))
                means = df.groupby("Placed")["iq"].mean()
                means.plot(kind="bar", ax=ax, color=["#ef4444", "#10b981"])
                ax.set_title("Average IQ: Placed vs Not Placed", fontsize=14, fontweight='bold')
                ax.set_ylabel("Average IQ", fontsize=12)
                ax.set_xlabel("Placement Status", fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                return {
                    "type": "plot",
                    "caption": "🧠 Avg IQ comparison plot",
                    "figure": fig,
                }

        except Exception as e:
            return {"type": "dataframe", "response": f"⚠️ Error analyzing data: {e}"}

    # -------------------------
    # Default Chat flow
    # -------------------------
    try:
        if llm is None:
            return {"type": "chat", "response": "⚠️ LLaMA model not loaded, cannot answer."}

        prompt = f"You are a helpful AI assistant specializing in career guidance and placement predictions.\nUser: {user_input}\nAssistant:"
        response = llm(
            prompt,
            max_tokens=256,
            stop=["User:", "Assistant:"],
        )
        answer = response["choices"][0]["text"].strip()
        return {
            "type": "chat",
            "response": answer,
            "followups": [
                "Can you summarize the placement dataset?",
                "What is the placement percentage?",
                "Show me a scatter plot of CGPA vs IQ",
                "Which matters more: CGPA or IQ?"
            ]
        }
    except Exception as e:
        return {"type": "chat", "response": f"⚠️ LLaMA inference failed: {e}"}
