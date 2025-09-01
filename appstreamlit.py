import streamlit as st
import asyncio
import json
from techmate_agent import techmate_agent, TechMateOutput

st.set_page_config(page_title="🤖 TechMate Assistant", layout="wide")

st.title("🤖 TechMate – AI Troubleshooter")
st.markdown(
    "TechMate is an **agentic RAG-powered virtual assistant** that searches the web, "
    "retrieves relevant docs, and builds a structured troubleshooting plan for your issue."
)

# Sidebar
st.sidebar.header("🖥️ Device Context")
device = st.sidebar.text_input("Device", value="Windows laptop")
os_name = st.sidebar.selectbox("Operating System", ["Windows", "macOS", "Linux"], index=0)
symptoms = [s.strip() for s in st.sidebar.text_area("Symptoms (comma-separated)").split(",") if s.strip()]
constraints = [c.strip() for c in st.sidebar.text_area("Constraints (comma-separated)").split(",") if c.strip()]

# Main query input
query = st.text_input("🔎 Describe your issue:", placeholder="e.g., WiFi disconnects after sleep on Windows 11")

if st.button("🚀 Ask TechMate") and query:
    with st.spinner("🔍 Searching the web, retrieving snippets, and planning steps..."):
        try:
            plan: TechMateOutput = asyncio.run(
                techmate_agent(query, device=device, os_name=os_name, symptoms=symptoms, constraints=constraints)
            )

            st.success("✅ Troubleshooting plan generated!")

            # --- Display structured plan ---
            st.subheader("📌 Issue Summary")
            st.write(plan.issue_summary)

            st.subheader("🤔 Likely Causes")
            st.write(plan.likely_causes)

            st.subheader("📝 Plan Overview")
            st.write(plan.plan_overview)

            st.subheader("⚡ Quick Checks")
            st.write(plan.quick_checks)

            st.subheader("🔧 Troubleshooting Steps")
            for step in plan.steps:
                with st.expander(f"{step.id}: {step.title}"):
                    st.markdown(f"**Rationale:** {step.rationale}")
                    st.markdown(f"**Action:** {step.action}")
                    if step.commands:
                        st.code("\n".join(step.commands), language="bash")
                    st.markdown(f"**Expected Outcome:** {step.expect}")
                    if step.if_fails_next:
                        st.markdown(f"➡️ If fails, go to: {step.if_fails_next}")

            # st.subheader("🧪 Diagnostics to Collect")
            # st.write(plan.diagnostics_to_collect)

            # st.subheader("✅ Resolution Criteria")
            # st.write(plan.resolution_criteria)

            # st.subheader("📈 Escalation Criteria")
            # st.write(plan.escalation_criteria)

            # st.subheader("⚠️ Safety Notes")
            # st.write(plan.safety_notes)

            st.subheader("🔗 Sources")
            for src in plan.sources:
                st.markdown(f"- [{src}]({src})")

            st.subheader("📌 Assumptions")
            st.write(plan.assumptions)

            st.progress(plan.confidence)

        except Exception as e:
            st.error(f"❌ Error: {e}")
