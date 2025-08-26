import streamlit as st

st.title("🚀 Simple LRDBenchmark Test")
st.write("If you can see this, Streamlit is working!")

st.header("Test Components")
st.write("This is a simple test to verify Streamlit functionality.")

# Test basic components
if st.button("Click me!"):
    st.success("Button works!")

# Test sidebar
st.sidebar.header("Test Sidebar")
st.sidebar.write("Sidebar is working!")

# Test metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Test Metric 1", "100", "10")
with col2:
    st.metric("Test Metric 2", "200", "-5")
with col3:
    st.metric("Test Metric 3", "300", "20")

st.success("✅ Simple test app loaded successfully!")
