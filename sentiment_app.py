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

    input_filename = st.text_input("Enter the name of your .md comments file: ")
    output_filename = st.text_input("Enter the desired output filename: ", "sentiment_analysis.md")

    max_displayed = st.text_input("What is the maximum number of comments with their classification do you want displayed as a sample of this analysis: ", "5")

    if st.button("Run Analysis"): 
        with st.spinner("Analyzing sentiments..."):
            blocks = process_md_file(input_filename)

            nb_positive = 0
            nb_negative = 0
            nb_neutral = 0
            nb_total_comments = 0
            results = []  # Store results to display later

            progress_bar = st.progress(0)
            for i, block in enumerate(blocks):
                sentiment = classify_sentiment(block)

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

            # Save to output file
            with open(output_filename, "w", encoding="utf-8") as output_f:
                for block in blocks:
                    sentiment = classify_sentiment(block) 
                    output_f.write(f"Comment: {block} -> Sentiment: {sentiment}\n")

                output_f.write(f"\n\nResults Summary: We have {nb_positive} positives comments, {nb_negative} negative comments and {nb_neutral} neutral comments. The overall NPS Score is: {NPS_Score}\n")

if __name__ == '__main__':
    main()
