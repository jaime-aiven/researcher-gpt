import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
wintr_api_key = os.getenv("WINTR_API_KEY")

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
    # post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    # response = requests.post(post_url, headers=headers, data=data_json)

    # Send a POST request to Wintr
    post_url = f"https://chrome.browserless.io/content?token={wintr_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


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


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are urls of relevant links & articles, you will scrape them to gather more information
            3/ You should not make things up, you should only write facts & data that you have gathered
            4/ Your research is not complete until you are sure your output complies will all the instructions below
            5/ Your output must contain the following sections: #Summary on the research target, #Summary of existing cloud technology stack, #Business Value Drivers, #Aiven Unique Capabilities, #Discovery Questions, #Sample cold email and #Sources, in this order.
            5/ Your output must contain the following sections: #Summary on the research target, #Summary of existing cloud technology stack, #Business Value Drivers, #Aiven Unique Capabilities, #Discovery Questions, #Sample cold email and #Sources, in this order.
            7/ Your output must contain insights on what topics, tone and keywords this person would be most receptive to in a cold email about AI cloud data infrastructure
            8/ The output should contain suggestions on how the Aiven data platform (which provides Kafka, Flink, PostgreSQL, MySQL, Cassandra, OpenSearch, CLickhouse, Redis, Grafana) in all major clouds) could address their needs for streaming, storing and serving data in the cloud. The emphasis is on a provocative point of view.
            9/ Your output must not list all the products that Aiven offers, but rather only the ones that would match the business value drivers of the company
            10/ The output should help a seller understand the target's problem, the monetary cost of the problem to their business, the solution to the problem, the $$ value of solving the problem , what $ they are prepared to spend to solve the problem, and the fact that Aiven can solve the problem
            11/ As the final part of the output, please write a sample 3-paragraph cold email to the research target from an Aiven seller that would address the pains uncovered from the provocative sales point of view of Aiven, in a way that maximizes the likelihood they engage in a sales conversation with Aiven.
            12/ The email should reference the technology that they already use and how Aiven can provide superior time to value with an unified platform, unmatched cost control and compliance by default.
            13/ In the final output, You should include all reference data & links to back up your research
            14/ Your output must be nicely formatted with headers for each section and bullet points. """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1500)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# This function extracts the paragraphs as sections.
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



# 4. Use streamlit to create a web app


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

        result = agent({"input": query})

        result_text = process_input(result['output'])
        
        # st.info(result['output'])

         # Define tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Summary",
            "Cloud technology stack",
            "Business Value Drivers",
            "Aiven Unique Capabilities",
            "Discovery Questions",
            "Sample cold email",
            "Sources"
        ])


        # Create and write tabs
        with tab1:
            st.write(remove_first_two_lines(result_text['summary']))
        
        with tab2:
            st.write(remove_first_two_lines(result_text['cloud_stack']))
        
        with tab3:
            st.write(remove_first_two_lines(result_text['value_drivers']))
        
        with tab4:
            st.write(remove_first_two_lines(result_text['aiven_capabilities']))
        
        with tab5:
            st.write(remove_first_two_lines(result_text['discovery_questions']))
        
        with tab6:
            st.write(remove_first_two_lines(result_text['cold_email']))
        
        with tab7:
            st.write(remove_first_two_lines(result_text['sources']))


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
