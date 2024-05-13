import os
from dotenv import load_dotenv
import pyperclip
import clipboard
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_verbose
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, create_structured_chat_agent
# from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
# from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st
#import re



load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
wintr_api_key = os.getenv("WINTR_API_KEY")


# Vectorize and store the Aiven 2000 csv data
loader = CSVLoader(file_path="A2Kfirmographics.csv", encoding="utf8")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)



#Set Verbose to true
set_verbose(True)

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# 2. Tool to search Stackshare

def stack_search(company_name):
    url = "https://google.serper.dev/search"

    # Append site:stackshare.com to the query to restrict results to StackShare
    full_query = f"site:stackshare.io {company_name}"

    payload = json.dumps({
        "q": full_query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(remove_multiple_line_breaks(response.text))

    return remove_multiple_line_breaks(response.text)


# Tool to remove double line breaks from scraping
def remove_multiple_line_breaks(text):
    # Normalize the line breaks from '\\n' to '\n'
    normalized_text = text.replace('\\n', '\n')

    # Initialize the variable for processing
    previous_text = normalized_text
    while True:
        # Replace two or more consecutive line breaks with a single line break
        current_text = re.sub(r'\n{2,}', '\n', previous_text)
        # Check if any more replacements are needed
        if current_text == previous_text:
            break
        previous_text = current_text
    
    return current_text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request to Browserless
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = remove_multiple_line_breaks(soup.get_text())
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")



# Tool to summarize longer texts

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")





# Function for similarity search

def retrieve_relevant_info_from_db(query):
    similar_data = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_data]

    # For debug
    # print(page_contents_array)

    return page_contents_array




# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(), Tool(
        name="Stack_search",
        func=stack_search,
        description="Useful for answering questions about a company's technology stack. You should ask targeted questions"
    )
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are urls of relevant links & articles, you will scrape them to gather more information
            3/ You should not make things up, you should only write facts & data that you have gathered
            4/ Your research is not complete until you are sure your output complies will all the instructions below
            5/ Your output must contain the following sections with these exact section names in order: Summary on the research target, Summary of existing cloud stack, Business Value Drivers, Aiven Unique Capabilities, Discovery Questions, Sample cold email and Sources, in this order.
            5/ Your output must contain the following sections with these exact section names in order: Summary on the research target, Summary of existing cloud stack, Business Value Drivers, Aiven Unique Capabilities, Discovery Questions, Sample cold email and Sources, in this order.
            7/ Your output must contain insights on what topics, tone and keywords this person would be most receptive to in a cold email about AI cloud data infrastructure
            8/ The output should contain suggestions on how the Aiven data platform (which provides Kafka, Flink, PostgreSQL, MySQL, Cassandra, OpenSearch, CLickhouse, Redis, Grafana) in all major clouds) could address their needs for streaming, storing and serving data in the cloud. The emphasis is on a provocative point of view.
            9/ Your output must not list all the products that Aiven offers, but rather only the ones that would match the business value drivers of the company
            10/ The output should help a seller understand the target's problem, the monetary cost of the problem to their business, the solution to the problem, the $$ value of solving the problem , what $ they are prepared to spend to solve the problem, and the fact that Aiven can solve the problem
            11/ As the final part of the output, please write a sample 3-paragraph cold email to the research target from an Aiven seller that would address the pains uncovered from the provocative sales point of view of Aiven, in a way that maximizes the likelihood they engage in a sales conversation with Aiven.
            12/ The email should reference the technology that they already use (especially databases and streaming services) and how Aiven can provide superior time to value with an unified platform, unmatched cost control and compliance by default. You can use the Stack_search tool for this.
            13/ In the final output, You should include all reference data & links to back up your research
            14/ Your output must be nicely formatted with headers for each section and bullet points. 
            
            This is the person and company you must research: {message}

Here is the data we have on the target company and similar ones: {firmographic_data}"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}


# Set up LLM with temperature 0 for most predictable results
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1500)


# ####### From RAG project, untested

# prompt = PromptTemplate(
#     input_variables=["message", "firmographic_data"],
#     template=system_message
# )


# chain = LLMChain(llm=llm, prompt=prompt)

# #################

# Legacy (works!)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# Retrieval augmented generation
def generate_response(message):
    
    # Use the RAG retrieval to get contextually relevant company data
    firmographic_data = retrieve_relevant_info_from_db(message)

    # Update the memory with the retrieved context
    memory.update(firmographic_data)

    # Configure agent to pass new prompt with retrieved context
    agent.system_message.content += f"\n\nHere is the data we have on the target company and similar ones: {firmographic_data}"
    

    response = agent({"input": message, "memory": firmographic_data})
    
    return response



# Extract the paragraphs as sections.
def process_input(text):
    sections = text.split('#')  # Assume the paragraph uses double newlines to separate sections.
    return {
        "start": sections[0] if len(sections) > 0 else "",
        "summary": sections[1] if len(sections) > 1 else "",
        "cloud_stack": sections[2] if len(sections) > 2 else "",
        "value_drivers": sections[3] if len(sections) > 3 else "",
        "aiven_capabilities": sections[4] if len(sections) > 4 else "",
        "discovery_questions": sections[5] if len(sections) > 5 else "",
        "cold_email": sections[6] if len(sections) > 6 else "",
        "sources": sections[7] if len(sections) > 7 else ""
        }


# This function removes the first two lines of each section (section titles, which are needed by the LLM but useless for the end human user)
def remove_first_two_lines(text):
    lines = text.split('\n')
    if len(lines) > 2:
        return '\n'.join(lines[2:])
    return ''  # Return an empty string if there are less than two lines


# This function should provide more accurate parsing of the sections
import re

def parse_llm_output(text):
    # Define the section headers based on the ## pattern observed
    headers = [
        "Summary on the research target",
        "Summary of existing cloud stack",
        "Business Value Drivers",
        "Aiven Unique Capabilities",
        "Discovery Questions",
        "Sample cold email",
        "Sources"
    ]

    # Escape headers to safely use them in a regex pattern
    escaped_headers = [re.escape(header) for header in headers]

    # Create a dictionary to store the sections
    sections = {}
    
    # Split the text at each header, keep the headers as delimiters
    pattern = r'(' + '|'.join(escaped_headers) + r')'
    parts = re.split(pattern, text)
    
    # The split includes headers as separate parts, so pair headers with their following content
    for i in range(1, len(parts), 2):  # Start from 1 and take steps of 2 to get headers
        if i + 1 < len(parts):
            sections[parts[i]] = parts[i + 1].strip()  # Strip whitespace and associate header with content

    return sections






# 4. Use streamlit to create a web app

def on_copy_click(text):
    st.session_state.copied.append(text)
    clipboard.copy(text)

if "copied" not in st.session_state: 
    st.session_state.copied = []

def main():
    st.set_page_config(page_title="Aiven AI PPoV prospecting agent", page_icon=":moneybag:", layout="wide")

    st.header(":crab: Aiven AI PPoV prospecting agent :moneybag: :crab:")
     

    st.write("The PPoV research takes about 1 minute to complete and accepts one target at a time.")
    
    query = st.text_input("""Enter research target (Full name and company):""")

    # if query:
    #     st.write("Researching ", query)

    #     result = agent({"input": query})

    #     st.info(result['output'])

    if query:
        st.write("Researching ", query)

        progress_bar = st.progress(0)
        # time.sleep(1)  # Simulate delay for fetching data
        progress_bar.progress(20)

        # result = agent({"input": query})
        result = generate_response({"input": query})


       
        sections = parse_llm_output(result['output'])
        tabs = st.tabs([k.replace("#", "") for k in sections.keys()])  # Create tabs without the '#' in the title

        for tab, key in zip(tabs, sections.keys()):
            with tab:
                st.write(sections[key])


        # # Create and write tabs
        # with tab1:
        #     st.write(remove_first_two_lines(result_text['summary']))
        
        # with tab2:
        #     st.write(remove_first_two_lines(result_text['cloud_stack']))
        
        # with tab3:
        #     st.write(remove_first_two_lines(result_text['value_drivers']))
        
        # with tab4:
        #     st.write(remove_first_two_lines(result_text['aiven_capabilities']))
        
        # with tab5:
        #     st.write(remove_first_two_lines(result_text['discovery_questions']))
        
        # with tab6:
        #     # st.button("Copy to clipboard 📋", on_click=on_copy_click, args=(remove_first_two_lines(result_text['cold_email']),))
        #     st.write(remove_first_two_lines(result_text['cold_email']))
            
        #     for text in st.session_state.copied:
        #         st.toast(f"Copied to clipboard: {text}", icon='✅' )
        
        # with tab7:
        #     st.write(remove_first_two_lines(result_text['sources']))


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content
