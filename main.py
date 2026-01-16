#3rd and final step using streamlit
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load model and data
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open('models/trained_model.pkl', 'rb'))
        vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        resumes_df = pd.read_csv('data/resumes_clean.csv')
        labels_df = pd.read_csv('data/labels_clean.csv')
        return model, vectorizer, resumes_df, labels_df
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.stop()

# Calculate match score using keyword matching (more generous than TF-IDF)
def calculate_match_score(job_desc, resume_text, vectorizer):
    try:
        # Convert to lowercase for comparison
        job_desc_lower = job_desc.lower()
        resume_lower = resume_text.lower()
        
        # Extract keywords from job description (words > 3 chars)
        job_keywords = set([word for word in job_desc_lower.split() if len(word) > 3])
        resume_words = set(resume_lower.split())
        
        # Calculate keyword match percentage
        if len(job_keywords) == 0:
            return 50.0  # Default score if no keywords
        
        matches = job_keywords.intersection(resume_words)
        keyword_score = (len(matches) / len(job_keywords)) * 100
        
        # Also use TF-IDF for additional scoring
        try:
            job_vec = vectorizer.transform([job_desc])
            resume_vec = vectorizer.transform([resume_text])
            tfidf_similarity = cosine_similarity(job_vec, resume_vec)[0][0]
            tfidf_score = tfidf_similarity * 100
        except:
            tfidf_score = 0
        
        # Combine both scores: 60% keyword match + 40% TF-IDF
        final_score = (keyword_score * 0.6) + (tfidf_score * 0.4)
        
        # Boost score slightly for category relevance (base score of 35)
        final_score = final_score * 0.65 + 35
        
        return max(0, min(100, final_score))
    except Exception as e:
        return 50.0  # Default fallback score

# Set page config
st.set_page_config(
    page_title="Company Recruitment Portal",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0F1419;
        color: #FFFFFF;
    }
    .main {
        background-color: #0F1419;
    }
    .stMetric {
        background-color: #1A1F2E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2D3748;
    }
    .rank-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <h1 style='text-align: center; color: #00D4FF;'>👔 Company Recruitment Portal</h1>
    <p style='text-align: center; color: #A0AEC0;'>Candidate Search & Ranking System</p>
    <p style='text-align: center; color: #718096; font-size: 12px;'>Powered by AI-ML | 2,484 Resume Database | 24 Job Categories</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Load resources
try:
    model, vectorizer, resumes_df, labels_df = load_resources()
except:
    st.error("Failed to load model and data files")
    st.stop()

# Define job categories
job_categories = [
    "ACCOUNTANT",
    "ADVOCATE",
    "AGRICULTURE-MANAGEMENT",
    "APPAREL",
    "ARTS",
    "AUTOMOBILE",
    "AVIATION",
    "BANKING",
    "BPO",
    "BUSINESS-DEVELOPMENT",
    "CHEF",
    "CONSTRUCTION",
    "CONSULTANT",
    "DESIGNER",
    "DIGITAL-MEDIA",
    "ENGINEERING",
    "FINANCE",
    "HEALTHCARE",
    "HR",
    "INFORMATION-TECHNOLOGY",
    "LAW",
    "MEDIA-ENTERTAINMENT",
    "SALES",
    "TEACHER"
]

# Sidebar for controls
with st.sidebar:
    st.markdown("### 📋 Search Parameters")
    
    # Category selection
    selected_category = st.selectbox(
        "Select Job Category",
        options=job_categories,
        help="Choose the job category to search resumes"
    )
    
    st.markdown("---")
    
    # Job description
    st.markdown("### 📝 Job Description")
    
    job_description = st.text_area(
        "Enter job description (optional but recommended)",
        value=f"Looking for a {selected_category.lower().replace('-', ' ')} with relevant experience and skills",
        height=250,
        key="job_desc"
    )
    
    # Number of results
    num_results = st.slider(
        "Number of candidates to display",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    st.markdown("---")
    
    # Database info
    st.markdown("### 📊 Database Info")
    
    category_count = len(resumes_df[resumes_df['category'] == selected_category])
    st.write(f"✅ Status: Active")
    st.write(f"📂 Selected Category: {selected_category}")
    st.write(f"👥 Resumes in Category: {category_count}")
    st.write(f"📦 Total Database: 2,484")
    st.write(f"🏢 Job Categories: 24")

# Main content
st.subheader(f"🔍 Search Results: {selected_category}")

search_button = st.button(
    "🔍 Search Database",
    use_container_width=True,
    type="primary"
)

if search_button:
    if not job_description.strip():
        st.error("❌ Please enter a job description")
    else:
        # Filter resumes by selected category
        category_resumes = resumes_df[resumes_df['category'] == selected_category].copy()
        
        if len(category_resumes) == 0:
            st.error(f"❌ No resumes found for {selected_category}")
        else:
            # Calculate match scores
            candidates = []
            
            with st.spinner("Calculating match scores..."):
                for idx, row in category_resumes.iterrows():
                    resume_text = str(row['resume_text'])
                    
                    if len(resume_text.strip()) > 0:
                        score = calculate_match_score(job_description, resume_text, vectorizer)
                        
                        candidates.append({
                            'rank': 0,
                            'candidate_id': idx,
                            'score': score,
                            'resume_text': resume_text,
                            'category': row['category']
                        })
            
            # Sort by score
            candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
            
            # Add rank
            for idx, candidate in enumerate(candidates_sorted[:num_results], 1):
                candidate['rank'] = idx
            
            candidates_display = candidates_sorted[:num_results]
            
            # Display summary
            st.success(f"✅ Found {len(category_resumes)} resumes in {selected_category}")
            st.info(f"🏆 Displaying top {len(candidates_display)} candidates")
            
            st.markdown("---")
            
            # Display leaderboard
            for candidate in candidates_display:
                rank = candidate['rank']
                score = candidate['score']
                resume_preview = candidate['resume_text']
                
                # Determine eligibility status
                if score >= 60:
                    rating = "🟢 Highly Eligible"
                    color = "#10B981"
                    border_color = "10B981"
                elif score >= 40:
                    rating = "🟡 Eligible"
                    color = "#F59E0B"
                    border_color = "F59E0B"
                else:
                    rating = "🔴 Not Eligible"
                    color = "#EF4444"
                    border_color = "EF4444"
                
                # Medal emoji
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                
                # Create card
                with st.container():
                    col_medal, col_info, col_score = st.columns([0.5, 2, 1.5])
                    
                    with col_medal:
                        st.markdown(f"<h2 style='text-align: center; color: {color};'>{medal}</h2>", unsafe_allow_html=True)
                    
                    with col_info:
                        st.markdown(f"""
                            <div style='padding: 10px;'>
                            <p style='margin: 0; font-weight: bold; color: white;'>Candidate #{rank}</p>
                            <p style='margin: 5px 0 0 0; color: {color};'>{rating}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_score:
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px; background: {color}20; 
                            border-radius: 8px; border: 2px solid {color};'>
                            <p style='margin: 0; color: {color}; font-weight: bold; font-size: 18px;'>{score:.1f}%</p>
                            <p style='margin: 5px 0 0 0; color: {color}; font-size: 12px;'>Match Score</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Expandable details
                    with st.expander(f"View Resume - Candidate #{rank}", expanded=(rank == 1)):
                        
                        col_detail1, col_detail2, col_detail3 = st.columns(3)
                        
                        with col_detail1:
                            st.metric("Match %", f"{score:.1f}%")
                        
                        with col_detail2:
                            st.metric("Rank", f"#{rank}")
                        
                        with col_detail3:
                            st.metric("Category", selected_category)
                        
                        # Resume preview
                        st.markdown("**Resume Content:**")
                        preview_length = 500
                        preview = resume_preview[:preview_length] + "..." if len(resume_preview) > preview_length else resume_preview
                        st.text_area(
                            label="Resume Text",
                            value=preview,
                            height=200,
                            disabled=True,
                            key=f"resume_{rank}"
                        )
                        
                        # Action buttons
                        col_action1, col_action2, col_action3 = st.columns(3)
                        
                        with col_action1:
                            if st.button(f"✅ Shortlist", key=f"shortlist_{rank}"):
                                st.success(f"✅ Candidate #{rank} shortlisted for interview")
                        
                        with col_action2:
                            if st.button(f"⏳ Waitlist", key=f"waitlist_{rank}"):
                                st.info(f"⏳ Candidate #{rank} added to waitlist")
                        
                        with col_action3:
                            if st.button(f"❌ Reject", key=f"reject_{rank}"):
                                st.error(f"❌ Candidate #{rank} rejected")
                    
                    st.markdown("")
            
            # Summary statistics
            st.markdown("---")
            st.markdown("### 📊 Summary Statistics")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Resumes", len(category_resumes))
            
            with col_stat2:
                highly_eligible = sum(1 for c in candidates_display if c['score'] >= 60)
                st.metric("Highly Eligible", highly_eligible)
            
            with col_stat3:
                eligible = sum(1 for c in candidates_display if 40 <= c['score'] < 60)
                st.metric("Eligible", eligible)
            
            with col_stat4:
                not_eligible = sum(1 for c in candidates_display if c['score'] < 40)
                st.metric("Not Eligible", not_eligible)
            
            # Score distribution
            st.markdown("### 📈 Score Distribution (Top 10)")
            
            top_10_scores = [c['score'] for c in candidates_display[:10]]
            top_10_ranks = [f"#{c['rank']}" for c in candidates_display[:10]]
            
            score_df = pd.DataFrame({
                'Candidate': top_10_ranks,
                'Match Score': top_10_scores
            })
            
            st.bar_chart(score_df.set_index('Candidate'))

else:
    st.info("👇 Select a job category, enter job description, and click 'Search Database' to find matching candidates")

st.markdown("---")

# Footer
st.markdown("""
    <p style='text-align: center; color: #4B5563; font-size: 12px;'>
    Company Recruitment Portal © 2025 | AI-Powered Screening System<br>
    Searches 2,484 resumes across 24 job categories | Confidential for company use only
    </p>
    """, unsafe_allow_html=True)
