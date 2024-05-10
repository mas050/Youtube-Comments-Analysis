import re
import os
import streamlit as st

from groq import Groq

os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
                    humour, sarcasm and double meaning often used online by human. \
                    "
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        #model="llama3-70b-8192",
        #model = "mixtral-8x7b-32768",
        #model = "gemma-7b-it",
        #model = "whisper-large-v3",
        temperature=0,
        max_tokens=5,
        top_p=1
    )
    
    # Simple extraction (can be made more robust)
    sentiment = response.choices[0].message.content
    return sentiment



def process_md_file(filename):
    """Parses the .md file and extracts text blocks starting with '\*'"""

    text_blocks = []
    current_block = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('\\*'):  # Detect start of a block
                # Add previous block if it exists
                if current_block:
                    text_blocks.append(''.join(current_block))
                    current_block = []  # Reset for new block

                # Add the current line (without the '\*')
                current_block.append(line[2:].strip())
            else:
                current_block.append(line)

    # Add the last block
    if current_block:
        text_blocks.append(''.join(current_block))

    return text_blocks


def main():
    st.title("Sentiment Analysis Tool") 

    uploaded_file = st.file_uploader("Choose a .md file", type="md")
    output_filename = st.text_input("Enter the desired output filename: ", "sentiment_analysis.md")

    max_displayed = st.text_input("What is the maximum number of comments with their classification do you want displayed as a sample of this analysis: ", "5")

    if st.button("Run Analysis"): 
        if uploaded_file is not None:
            with st.spinner("Analyzing sentiments..."):
                file_content = uploaded_file.getvalue().decode("utf-8")
                blocks = process_md_file(file_content)
    
                nb_positive = 0
                nb_negative = 0
                nb_neutral = 0
                nb_total_comments = 0
                results = []  # Store results to display later
    
                progress_bar = st.progress(0)

                # Create output for download
                output_content = "# Sentiment Analysis Results\n\n" 

                for i, block in enumerate(blocks):
                    sentiment = classify_sentiment(block)
                    output_content += f"**Comment #{i+1}:**\n{block}\n**Sentiment:** {sentiment}\n\n"
    
                    if i+1 <= int(max_displayed):
                        results.append(f"Comment #{i+1}:\n\n {block} \n\n --> Sentiment: {sentiment}")  # Store for summary
    
                    if sentiment.lower() == "positive":
                        nb_positive += 1
                    elif sentiment.lower() == "negative":
                        nb_negative += 1
                    else:
                        nb_neutral += 1
    
                    progress_bar.progress((i + 1) / len(blocks))
    
                NPS_Score = round((nb_positive - nb_negative)/(nb_positive + nb_negative + nb_neutral) * 100)
    
                nb_total_comments=nb_positive+nb_negative+nb_neutral

                # Add summary results
                output_content += f"**Results Summary:**\n* Total Comments: {nb_total_comments}\n"
                output_content += f"* Positive: {nb_positive}\n* Negative: {nb_negative}\n* Neutral: {nb_neutral}\n"
                output_content += f"* NPS Score: {NPS_Score}\n\n"

    
                # Display summary FIRST
                st.success("Analysis complete!")
                st.write(f"Results Summary: \n\n We had a total of {nb_total_comments} comments in the original file --> {nb_positive} positive, {nb_negative} negative and {nb_neutral} neutral. \n\n The overall NPS Score is --> {NPS_Score} \n\n")
    
                # Now display detailed results
                # Detailed comments section (with custom styling)
                st.markdown("""
                <div style="background-color: #001f3f; color: white; padding: 10px; border-radius: 10px;">
                <h2 style="font-size: 22px;"> Detailed Comments Classification (Sample of {})</h2>
                </div>
                """.format(max_displayed), unsafe_allow_html=True)
    
                for result in results:
                    st.write("\n")
                    st.write(result)
    
                st.download_button(
                    label="Download Results",
                    data=output_content,
                    file_name=output_filename,
                    mime='text/markdown'
                )
        else:
            st.error("Please upload a .md file first.")


if __name__ == '__main__':
    main()


