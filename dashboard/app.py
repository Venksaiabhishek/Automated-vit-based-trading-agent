import streamlit as st
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

# Append src to path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.graph import agentic_system
from src.agent.state import AgentState

st.set_page_config(page_title="VIT-Enhanced Agentic Trading", layout="wide")

st.title("VIT-Enhanced Agentic Trading System")
st.markdown("### Committee of Goal-Oriented AI Agents")

# Sidebar setup
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
has_position = st.sidebar.checkbox("Simulate Existing Position", value=False)
sell_date = st.sidebar.text_input("Current Sell Target Date", value="2026-03-30") if has_position else "None"
dummy_chart_trigger = st.sidebar.button("Run Multi-Modal Analysis")

# Main Dashboard
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Market Perception")
    st.info("Upload or link a candlestick chart for the Vision Agent.")
    # Placeholder for chart image upload/display
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Candlestick_chart_scheme_03-en.svg/1024px-Candlestick_chart_scheme_03-en.svg.png", caption="Mock Live Chart", use_container_width=True)

with col2:
    st.subheader("Agent Reasoning Console")
    console_placeholder = st.empty()
    
    if dummy_chart_trigger:
        with st.spinner(f"Initiating Agentic Analysis for {ticker}..."):
            
            # Fetch live data and generate chart
            from src.data.ingestion import fetch_live_data, generate_candlestick_chart
            df = fetch_live_data(ticker, period="1mo", interval="1d")
            chart_path = generate_candlestick_chart(ticker, df, output_path=f"{ticker}_live_chart.png")
            
            if not chart_path:
                chart_path = "mock_chart.png" # fallback if generation fails
                
            # Initialize State
            initial_state: AgentState = {
                "ticker": ticker.upper(),
                "chart_image_path": chart_path,
                "vision_signal": "",
                "sentiment_signal": "",
                "memory_signal": "",
                "thought_process": [],
                "current_position": has_position,
                "entry_price": 100.0,
                "sell_target_date": sell_date,
                "strategist_decision": "",
                "strategist_confidence": 0.0,
                "final_execution_status": "",
                "final_quantity": 0.0
            }
            
            # Execute LangGraph Workflow
            try:
                final_output = agentic_system.invoke(initial_state)
                
                # Display output in a styled box
                logs = "\n".join([f"> {log}" for log in final_output['thought_process']])
                
                decision_color = "green" if "BUY" in final_output['strategist_decision'] else "red" if "SELL" in final_output['strategist_decision'] else "gray"
                
                st.markdown(f"### Final Strategist Thesis: <span style='color:{decision_color}'>{final_output['strategist_decision']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {final_output['strategist_confidence']:.2f} | **Status:** {final_output['final_execution_status']} | **Quantity:** {final_output['final_quantity']}")
                
                with st.expander("View Full Agent Debate Log", expanded=True):
                    st.code(logs, language="bash")
                    
            except Exception as e:
                st.error(f"Agent Execution Failed: {e}")
                st.info("Check if GEMINI_API_KEY and other environment variables are configured correctly.")
