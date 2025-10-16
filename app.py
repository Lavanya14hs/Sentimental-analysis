import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit default elements that might create blank space
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp > header {display: none;}
.css-18ni7ap {display: none;}
.css-hxt7ib {display: none;}
.css-1d391kg {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

/* Remove default streamlit styling and blank spaces */
.main > div.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}

/* Hide any default streamlit elements that create blank space */
.css-18ni7ap, .css-hxt7ib, .css-1d391kg {
    display: none !important;
}

/* Ensure no top margin/padding on main container */
div.stApp > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Global styles */
* {
    font-family: 'Inter', sans-serif !important;
}

/* Removed background styling - now uses default Streamlit background */
.stApp {
    min-height: 100vh;
}

/* App header bar */
.app-header {
    padding: 1rem 2rem;
    margin: 0;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 1000;
}

.app-header h2 {
    margin: 0 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.glass-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
    padding: 3rem;
    margin: 0 auto 2rem auto;
    max-width: 1200px;
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Hero title with gradient text */
.hero-title {
    font-size: 4rem !important;
    font-weight: 900 !important;
    text-align: center;
    background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    text-shadow: 0 10px 20px rgba(0,0,0,0.1);
    animation: textGlow 3s ease-in-out infinite alternate;
}

@keyframes textGlow {
    from { filter: drop-shadow(0 0 20px rgba(255, 107, 107, 0.5)); }
    to { filter: drop-shadow(0 0 40px rgba(78, 205, 196, 0.8)); }
}

/* Subtitle styling */
.hero-subtitle {
    font-size: 1.4rem;
    text-align: center;
    color: rgba(0, 0, 0, 0.7);
    margin-bottom: 3rem;
    line-height: 1.6;
    font-weight: 400;
}

/* Card styling with hover effects */
.feature-card {
    background: rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    padding: 2rem;
    margin-bottom: 2rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.1);
    background: rgba(0, 0, 0, 0.08);
}

/* Neon button styling */
.neon-button {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 1rem 3rem !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    width: 100%;
}

.neon-button:before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.neon-button:hover:before {
    left: 100%;
}

.neon-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(255, 107, 107, 0.6);
}

/* Input field styling */
.stTextInput > div > input {
    background: rgba(0, 0, 0, 0.05) !important;
    backdrop-filter: blur(10px);
    border: 2px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 15px !important;
    color: #333 !important;
    padding: 1rem !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease;
}

.stTextInput > div > input:focus {
    border-color: #ff6b6b !important;
    box-shadow: 0 0 20px rgba(255, 107, 107, 0.3) !important;
    background: rgba(0, 0, 0, 0.08) !important;
}

/* File uploader styling */
.stFileUploader > div {
    background: rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
    border: 2px dashed rgba(0, 0, 0, 0.2);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: #ff6b6b;
    background: rgba(0, 0, 0, 0.08);
}

/* Headers styling */
h1, h2, h3 {
    color: #333 !important;
    font-weight: 700 !important;
}

/* Dataframe styling */
.stDataFrame > div {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

/* Sentiment emoji animations */
.sentiment-result {
    background: rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
    border: 1px solid rgba(0, 0, 0, 0.1);
    animation: bounceIn 0.6s ease-out;
}

@keyframes bounceIn {
    0% { transform: scale(0.3); opacity: 0; }
    50% { transform: scale(1.05); }
    70% { transform: scale(0.9); }
    100% { transform: scale(1); opacity: 1; }
}

.sentiment-emoji {
    font-size: 4rem !important;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.sentiment-text {
    color: #333 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-top: 1rem;
}

/* Stats cards */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-card {
    background: rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    background: rgba(0, 0, 0, 0.08);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 900;
    color: #ff6b6b;
    display: block;
}

.stat-label {
    color: rgba(0, 0, 0, 0.7);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Chart container */
.chart-container {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 1rem;
    margin: 2rem 0;
}

/* Loading animation */
.loading {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
}

.loading-spinner {
    width: 50px;
    height: 50px;
    border: 5px solid rgba(0, 0, 0, 0.1);
    border-top: 5px solid #ff6b6b;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Success/Error messages */
.stSuccess, .stError, .stWarning {
    background: rgba(0, 0, 0, 0.05) !important;
    backdrop-filter: blur(15px);
    border-radius: 15px !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
}

/* Expander styling */
.stExpander {
    background: rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    border: 1px solid rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Mock functions for missing dependencies - replace with actual implementations
@st.cache_resource
def load_model_artifacts():
    # Placeholder - replace with your actual model loading
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, vectorizer, label_encoder
    except:
        # Mock objects for demo purposes
        class MockModel:
            def predict(self, X): return [0] * len(X)
        class MockVectorizer:
            def transform(self, X): return [[0] * 100] * len(X)
        class MockEncoder:
            classes_ = ['positive', 'negative', 'neutral']
            def inverse_transform(self, y): return [self.classes_[0] for _ in y]
        
        return MockModel(), MockVectorizer(), MockEncoder()

model, vectorizer, label_encoder = load_model_artifacts()

def preprocess_text_with_intensifiers_negations(text):
    intensifiers = ['very', 'really', 'extremely', 'so', 'too']
    negations = ['not', 'no', 'never', 'none', 'cannot']
    text = str(text).lower()
    tokens = text.split()
    new_tokens = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in negations and i + 1 < len(tokens):
            new_tokens.append(token + '_' + tokens[i + 1])
            skip_next = True
        elif token in intensifiers and i + 1 < len(tokens):
            new_tokens.append(token + '_' + tokens[i + 1])
            skip_next = True
        else:
            new_tokens.append(token)
    return ' '.join(new_tokens)

def clean_text_basic(text):
    try:
        custom_stopwords = set(stopwords.words('english'))
    except:
        custom_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    text = str(text).encode('ascii', 'ignore').decode()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

def sentiment_emoji(sentiment):
    emoji_map = {
        "positive": "😃",
        "negative": "😞", 
        "neutral": "😐"
    }
    return emoji_map.get(sentiment.lower(), "🤖")

def create_sentiment_chart(sentiment_counts):
    """Create a beautiful interactive chart for sentiment distribution"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(sentiment_counts.index),
            y=list(sentiment_counts.values),
            marker=dict(
                color=['#4CAF50', '#FF5722', '#FFC107'],
                line=dict(color='rgba(0,0,0,0.8)', width=2)
            ),
            text=[f"{sentiment_emoji(s)}<br>{c}" for s, c in zip(sentiment_counts.index, sentiment_counts.values)],
            textposition='auto',
            textfont=dict(size=14, color='black')
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Sentiment Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'black'}  # Corrected here
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Sentiment',
            tickfont=dict(color='black'),  # Corrected here
            gridcolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            title='Number of Reviews',
            tickfont=dict(color='black'),  # Corrected here
            gridcolor='rgba(0,0,0,0.2)'
        ),
        height=400
    )
    
    return fig


if 'page' not in st.session_state:
    st.session_state.page = 'intro'

def show_intro():
    # Add app header
    st.markdown('''
    <div class="app-header">
        <h1>Multi Platform Social Media  Sentiment Analysis</h1>
    </div>
    ''', unsafe_allow_html=True)
    
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="feature-card">
            <h3>🎯 Precision Analysis</h3>
            <p>Logistic regression algorithms trained for sentiment detection. 
            If words or phrases are unfamiliar or unseen during training, the model conservatively 
            classifies them as neutral.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="feature-card">
            <h3>📊 Beautiful Visualizations</h3>
            <p>Interactive charts and graphs that bring your data to life with stunning visuals</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="feature-card">
            <h3>⚡ Lightning Fast</h3>
            <p>Process thousands of reviews in seconds with optimized performance</p>
        </div>
        ''', unsafe_allow_html=True)
    
  
    
    if st.button("🚀 Launch Sentiment Analysis", key="launch_btn"):
        st.session_state.page = 'tool'
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_tool():
    # Add app header
    st.markdown('''
    <div class="app-header">
        <h1>Multi Platform Social Media  Sentiment Analysis</h1>
    </div>
    ''', unsafe_allow_html=True)
    
    
    # Header with back button
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back", key="back_btn"):
            st.session_state.page = 'intro'
            st.rerun()
    with col2:
        st.markdown('<h3 style="margin-top: 0;"> Sentiment Analysis Dashboard</h3>', unsafe_allow_html=True)
    
    
    st.markdown("### 📁 Batch Analysis: Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Drop your CSV file here or click to browse(Your CSV file should contain a 'Text' column with the reviews to analyze)", 
        type="csv",
        help="Your CSV file should contain a 'Text' column with the reviews to analyze"
    )
    
    if uploaded_file:
        try:
            # Show loading animation
            with st.spinner('🔮 Analyzing your data with AI magic...'):
                df = pd.read_csv(uploaded_file)
                
                if 'Text' not in df.columns:
                    st.error("❌ CSV must include a 'Text' column.")
                else:
                    st.success(f"✅ Successfully loaded {len(df)} reviews!")
                    
                    # Show preview
                    with st.expander("👀 Preview of your data"):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process data
                    df['clean_text'] = df['Text'].astype(str).apply(preprocess_text_with_intensifiers_negations)
                    df['clean_text'] = df['clean_text'].apply(clean_text_basic)
                    X = vectorizer.transform(df['clean_text'])
                    preds = model.predict(X)
                    df['Sentiment'] = label_encoder.inverse_transform(preds)
                    
                    # Calculate statistics
                    sentiment_counts = df['Sentiment'].value_counts().reindex(
                        label_encoder.classes_, fill_value=0
                    )
                    total = sentiment_counts.sum()
                    
                    # Statistics cards
                    st.markdown("### 📊 Analysis Results")
                    
                    cols = st.columns(len(label_encoder.classes_))
                    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
                        with cols[i]:
                            percentage = (count / total * 100) if total > 0 else 0
                            st.markdown(f'''
                            <div class="stat-card">
                                <div style="font-size: 3rem;">{sentiment_emoji(sentiment)}</div>
                                <span class="stat-number">{count}</span>
                                <span class="stat-label">{sentiment}<br>({percentage:.1f}%)</span>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # Interactive chart
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    chart = create_sentiment_chart(sentiment_counts)
                    st.plotly_chart(chart, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Detailed results
                    with st.expander("📋 Detailed Results"):
                        st.dataframe(df[['Text', 'Sentiment']], use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Results",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Analysis Sectiona

    st.markdown("### ⚡ Quick Sentiment Check")
    st.markdown("Type or paste any text below to get instant sentiment analysis!")
    
    user_text = st.text_area(
        "Enter your text here...", 
        height=100,
        placeholder="Example: This product is absolutely amazing! I love it so much."
    )
    
    if st.button("🎯 Analyze Sentiment", key="analyze_btn"):
        if user_text.strip():
            with st.spinner('🤖 AI is thinking...'):
                # Process text
                processed = preprocess_text_with_intensifiers_negations(user_text)
                processed = clean_text_basic(processed)
                X = vectorizer.transform([processed])
                pred = model.predict(X)
                sentiment = label_encoder.inverse_transform(pred)[0]
                
                # Show result with animation
                st.markdown(f'''
                <div class="sentiment-result">
                    <div class="sentiment-emoji">{sentiment_emoji(sentiment)}</div>
                    <div class="sentiment-text">Sentiment: {sentiment.upper()}</div>
                    <p style="color: rgba(0,0,0,0.7); margin-top: 1rem;">
                        Our AI has analyzed your text and determined the emotional tone!
                    </p>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("⚠ Please enter some text to analyze!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main app logic
if st.session_state.page == 'intro':
    show_intro()
else:
    show_tool()