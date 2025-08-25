Here’s a polished version of your `README.md` with better formatting and readability, while keeping **all your original content exactly as written** (no wording changes, only structural and stylistic improvements):

# Introduction to the project

Digico is a forward-thinking startup that provides a wide range of solutions, from cutting-edge AI products to robust cloud services to help customers achieve cloud transformation and AI integration.

Given the diversity of these offerings, we created **DigiBot** — an intelligent chatbot designed to help customers navigate Digico’s ecosystem. DigiBot not only introduces customers to the company, the team, and our accumulated expertise, but also guides them toward solutions that best fit their unique needs.


# Structure of the repository


/
├── Scraping\_Digico\_Website/
│   ├── scraped\_data/
│   │   ├── cleaned\_text\_content.txt (2)
│   │   └── crawling\_raw\_output.txt (1)
│   └── scraping\_pipeline.ipynb   # pipeline to scrape the website
├── requirements.txt
├── README.md
├── .env                          # environmental variables
└── .gitignore


**(1)** : a text file that contains the scraped text data with the links extracted from each page + additional tags that help delimiting each page extracted info  

**(2)** : A text file that contains only the text information extracted from each page after removing the following:

- page headers  
- section markers  
- links  
- the ☁️ emoji (which is a frequent emoji across the website)  
- adding "." to the end of each paragraph (this will help with chunking)


# Phase 1: Scraping the data from Digico's website

## Step 1: Scraping Digico's website

### scraping_pipeline.ipynb

This Jupyter notebook contains a complete pipeline for scraping Digico's website in order to get the most valuable information from it.  

The pipeline has the following steps:

- makes requests that appear to come from a legitimate Chrome browser  
- scrapes the page based on the sections so the text inside of each section will be agglomerated and near together to retain semantics and structure of the page  
- gets rid of anchors that retrieve redundant and repetitive information  
- gets rid of hidden elements  
- gets rid of unwanted selectors (parts that provide information with no important meaning)  
- excludes nav and footer text since they present no valuable information (but keeps links from these parts)  
- replaces Elementor counters with final values  
- extracts links from each page  
- implements a recursive crawling strategy based on the DFS algorithm for preserving semantic context  


## Step 2: Cleaning the scraped data

The file **crawling_raw_output.txt** is of the following structure:


\=== PAGE: [https://digico.solutions/ai-assessment/](https://digico.solutions/ai-assessment/) ===
\=== TEXT ===
AI Readiness Assessment Discover Your Organization’s AI Potential
Unlock ....

\=== LINKS ===
[https://digico.solutions/](https://digico.solutions/)

- `=== TEXT ===` section contains the text extracted from each page  
- `=== LINKS ===` section contains the links extracted from each page  

So we need to clean the structure to pass it to the chunking pipeline by removing:  

- page headers  
- section markers  
- links  
- the ☁️ emoji (which is a frequent emoji across the website)  
- adding "." to the end of each paragraph (this will help with chunking)

So in **cleaned_text_content.txt** you will only find the text information extracted from each page.  


# How to use the repository

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
````

Then open **Scraping\_Digico\_Website/scraping\_pipeline.ipynb** and run all the code cells.

