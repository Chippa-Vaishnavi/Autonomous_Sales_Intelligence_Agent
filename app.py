import logging
import streamlit as st
from agents.supervisor_agent import SupervisorAgent
from utils.loader import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Autonomous Sales Intelligence Agent', page_icon='📈', layout='wide')

st.title('Autonomous Sales Intelligence Agent')
st.markdown('A production-ready sales analytics and generative AI system for executive insights, forecasting, and conversational BI.')

uploaded_file = st.file_uploader('Upload your sales dataset (CSV)', type=['csv'])

if uploaded_file is not None:
    df = load_dataset(uploaded_file)
    supervisor = SupervisorAgent(df)

    st.sidebar.header('Navigation')
    page = st.sidebar.selectbox('Choose view', [
        'Dashboard',
        'Dataset Profile',
        'Sales Analytics',
        'Forecasting',
        'AI Insights',
        'Recommendations',
        'Chat with Data',
        'Download Report'
    ])

    if page == 'Dashboard':
        supervisor.render_dashboard()
    elif page == 'Dataset Profile':
        supervisor.render_profile()
    elif page == 'Sales Analytics':
        supervisor.render_sales_analytics()
    elif page == 'Forecasting':
        supervisor.render_forecasting()
    elif page == 'AI Insights':
        supervisor.render_insights()
    elif page == 'Recommendations':
        supervisor.render_recommendations()
    elif page == 'Chat with Data':
        supervisor.render_chat()
    elif page == 'Download Report':
        supervisor.render_report()
else:
    st.info('Upload the cleaned sales CSV file to start the Autonomous Sales Intelligence Agent.')
