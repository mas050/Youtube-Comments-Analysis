import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import time
import os
from groq import Groq

# --- Configuration ---
API_KEY = "AIzaSyBy9sh953p-T_uaSODmHf_vu5VG3EDLqVo"  # st.secrets["YOUTUBE_API_KEY"]
os.environ["GROQ_API_KEY"] = "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"  # st.secrets["GROQ_API_KEY"]
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Helper Functions ---
def classify_sentiment(comment):
    """Classifies a comment's sentiment using LLAMA 3"""
    prompt = f"Classify this user comment with one and only one of the following response answer: Positive, Negative, Neutral.  \
        Only output one of the allowed word based on the full context and deep meaning of the comment. Here's the comment to classify: {comment} \
        Do not ouput anything else other than: Positive, Negative, Neutral."

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a social media expert in classifing comments received on social media pages. \
                    You always classify the sentiment of user comments using one and only one of the following options: positive, negative or neutral. \
                    To do the classification, you always make sure to understand and analyze the full comment and do not just based your classification on one word within the comment. You perfectly understand \
                    humour, sarcasm and double meaning often used online by human."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=5,
        top_p=1
    )
    
    # Simple extraction (can be made more robust)
    sentiment = response.choices[0].message.content
    return sentiment


def summarize_comments(comments):
    """Summarizes a list of comments using Groq LLM API"""
    prompt = "Summarize the following group of comments written by various viewers: " + ", ".join(comments)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a summarization expert. You always summarize the main points of a given text. \
                    You always make sure to understand the context and the main ideas of the text before summarizing it. \
                    You perfectly understand humour, sarcasm and double meaning often used online by human. \
                    If some viewers are proposing new ideas for next videos or providing recommendations on how to improve the content of the channel, please provide a summarized list. If no such comment is present\
                    please do not make up anything, you have to be factual!\
                    However, the most important task here is to provide a comprehensive summary of all the comments when regrouped together so it doesn't feel like reading individual comment one by one."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        temperature=0.8,
        max_tokens=250,
    )
    return response.choices[0].message.content


def get_video_comments(youtube, video_id, max_comments, max_retries=5):
    comments = []
    next_page_token = None
    retries = 0

    while retries < max_retries:
        try:
            response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=max(0, min(100, max_comments - len(comments))),  # Dynamically adjust maxResults
                pageToken=next_page_token
            ).execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # No more pages

        except Exception as e:
            if "invalid page token" in str(e):
                retries += 1
                if retries < max_retries:
                    print(f"Invalid page token. Retrying in 5 seconds... ({retries}/{max_retries})")
                    time.sleep(5)
                else:
                    print("Maximum retries reached. Aborting.")
                    break  # Exit the loop
            else:
                raise e  # Re-raise the exception for other errors

    return comments[:max_comments]  # Trim comments if exceeding max_comments

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="YouTube Comment Analyzer", page_icon=":movie_camera:")
    st.title(":movie_camera: YouTube Comment Sentiment Analysis")

    # Input and Parameters
    video_url = st.text_input("Enter YouTube Video URL:")
    max_comments = st.slider("Max comments to fetch:", 1, 100, 10)  # Improved slider for input
    max_displayed = st.slider("Max comments to display:", 1, max_comments, 5)

    # Analysis Trigger
    if st.button("Analyze Comments"):
        if video_url:
            with st.spinner("Working on it... This may take a few minutes."):
                try:
                    # 1. Scrape Comments
                    youtube = build("youtube", "v3", developerKey=API_KEY)
                    video_id = video_url.split("v=")[-1]
                    comments = get_video_comments(youtube, video_id, max_comments) 
                    
                    if not comments:
                        st.warning("No comments found or an unexpected error occurred. Please try again.")
                        return

                    df = pd.DataFrame(comments, columns=["Comment"])

                    # 2. Sentiment Analysis with Progress Bar
                    progress_bar = st.progress(0)  # Ensure progress bar is initialized
                    for i, row in df.iterrows():
                        df.at[i, "Sentiment"] = classify_sentiment(row["Comment"])
                        if progress_bar:  # Check if progress_bar is not None
                            progress_bar.progress(int((i + 1) / len(df) * 100))

                    # 3. Calculate Metrics:
                    nb_positive = (df['Sentiment'] == 'Positive').sum()
                    nb_negative = (df['Sentiment'] == 'Negative').sum()
                    nb_neutral = (df['Sentiment'] == 'Neutral').sum()
                    nb_total_comments = len(df)
                    NPS_Score = round((nb_positive - nb_negative) / nb_total_comments * 100)

                    # 4. Display Summary:
                    st.success("Analysis complete!")
                    st.write(f"""
                    **Results Summary:**

                    - Total comments: {nb_total_comments}
                    - Positive: {nb_positive}
                    - Negative: {nb_negative}
                    - Neutral: {nb_neutral}
                    - NPS Score: {NPS_Score}
                    """)

                    # 5. Display Detailed Sample (using st.expander for better UX)
                    with st.expander("Detailed Comments (Sample)"):
                        st.dataframe(df[['Comment', 'Sentiment']].head(max_displayed))

                    # 6. Download Results (using st.download_button)
                    st.download_button(
                        label="Download Results",
                        data=df.to_csv(index=False),
                        file_name=f"{video_id}_sentiment_analysis.csv",
                        mime="text/csv"
                    )

                    # 7. Summarize Comments
                    summarized_comments = summarize_comments(df["Comment"].tolist())
                    st.write(f"""
                    **Summarized Comments:**

                    {summarized_comments}
                    """)

                except Exception as e:
                    st.error(f"Oops! Something went wrong: {e}")
        else:
            st.warning("Please enter a valid YouTube video URL.")


if __name__ == "__main__":
    main()
