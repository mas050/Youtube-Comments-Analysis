# Youtube Comments Analysis Tool with Streamlit and LLAMA 3

**Short Description:**

This Python project analyzes the sentiment (positive, negative, neutral) of user comments within .md files, leveraging the power of LLAMA 3 and displaying the results in an intuitive Streamlit web interface.

**Getting Started**

**Prerequisites:**

* Python 3.x
* Streamlit (`pip install streamlit`)
* Groq API Key (`https://groq.com/signup`)

**Installation:**

1. Clone this repository:
   ```bash 
   git clone https://github.com/mas050/Youtube-Comments-Analysis.git
   ```

2. Install required libraries:
   ```bash
   cd sentiment-analysis-tool
   pip install -r requirements.txt 
   ```

**Configuration:**

1. **Obtain a Groq API Key:** Create a free Groq account and get your API key.

2. **Set your API key:**
    *  Export your GROQ_API_KEY as an environment variable, replacing 'your_api_key':
       ```bash
       export GROQ_API_KEY=your_api_key
       ```

**Usage**

1. **Prepare your input file:** Create an .md file with your comments. Each comment block should start with `\*`.

2. **Run the script:**
   ```bash
   streamlit run sentiment_analysis.py
   ```

3. **Use the web interface:**
   * Enter the name of your input .md file.
   * Specify a name for the output file.
   * Control how many sample comments are displayed in the analysis.
   * Click "Run Analysis". 

**Output**

* The Streamlit web interface will display:
   * Summary of sentiment analysis results (number of positive, negative, neutral comments, and NPS Score).
   * A sample of analyzed comments with their classification.

* An output file (.md) will be created, containing detailed analysis results. 

**Key Features**

* Leverages LLAMA 3 for accurate sentiment classification of user comments.
* Processes comments from .md files.
* Provides a user-friendly Streamlit web interface.
* Calculates and displays the Net Promoter Score (NPS).
* Generates a comprehensive output file with results.

**Explanation**

The project incorporates the following key elements:

* **Import Statements:** Necessary libraries for text processing, LLAMA access, and Streamlit web app creation.
* **Groq Setup:** Connects to the Groq API using your provided API key.
* **`classify_sentiment` Function:**  Sends a comment to the LLAMA 3 model, instructing it to classify the sentiment and return the result.
* **`process_md_file` Function:** Parses an .md file, extracting comment blocks.
* **`main` Function:**  Defines the Streamlit interface with input fields, handles file processing, sentiment analysis, and displays results.

**Contributing**

Feel free to contribute to this project! Open issues, submit pull requests, and help enhance this sentiment analysis tool.

Let me know if you'd like any modifications to the README or additional sections. 
