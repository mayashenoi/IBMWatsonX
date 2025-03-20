# Generate Response word document for 3 clauses of Technical Criteria from RFP
# Imports
import pandas as pd
from docxtpl import DocxTemplate
from datetime import date, datetime

import os, getpass
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
import wget
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# IBM watsonx access
ibm_cloud_api_key=""
project_id=""
watson_url=""



credentials = Credentials(
    url=watson_url,
    api_key=ibm_cloud_api_key )
api_client = APIClient(credentials=credentials, project_id=project_id)

#Load knowledgebase into Chroma - the open-source embedding database.
#This is a sample knowledge text with only relevant text from 3 clauses being considered
#This should be multiple files of various content format which can be uploaded as knowledgebase in embeddings database
#load from local directory
filename = 'knowledge.txt'
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id
    )
docsearch = Chroma.from_documents(texts, embeddings)

model_id = ModelTypes.GRANITE_13B_INSTRUCT_V2

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
    }


watsonx_granite = WatsonxLLM(
   model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)
# Build prompt
template = """ Use the following pieces of context to answer the question at the end. You are a technical proposal writer, specific to engineering, procurement, and construction services.
 You have to explain how your company fulfils the enquired criteria in an RFP response. 
 Answer in detail how your company fulfils the criteria as mentioned in the question?
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa = RetrievalQA.from_chain_type(
    llm=watsonx_granite,
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

#These clauses have been extracted from a publically available RFP of Damodar Valley corporation
# we could extract all clauses from any RFP document using Docling or any document processor and rex
# One could create a dataframe of all the clauses and process them in a loop
# For the purpose of simplicity of the example, we are simply using text of the three clauses
# Clause 1.1.0
query=""" The Bidder should have executed on Engineering, Procurement and Construction (EPC)
basis, minimum one (1) no. Coal based/Lignite based power plant of atleast 500 MW Unit
capacity comprising of atleast a) Coal / lignite handling plant, b) Cooling Tower, c) Water /
waste water treatment plant or DM plant, including associated civil works, Structural and
Electrical systems for the above equipments and systems as a single package, which is in
successful operation for a period of not less than one (1) year prior to the date of techno-
commercial bid opening."""
result1 = qa.invoke(query)

# Clause 1.2.0
query=""" The Bidder (itself or along with its subsidiary(ies)) should have executed on Engineering,
Procurement and Construction (EPC) basis, minimum one (1) no. Coal based/Lignite based
power plant of installed capacity not less than 250 MW comprising of a) Main Power Plant and
b) Balance of Plant (having Coal / lignite handling plant, Cooling Tower, Water / waste water
treatment plant or DM plant) facilities, including associated civil works, Structural and Electrical
systems for the above equipments and systems as a single package, which is in successful
operation for a period of not less than one (1) year prior to the date of techno-commercial bid
opening."""
result2 = qa.invoke(query)

# Clause 1.3.0
query=""" Bidder shall be a Consortium (unincorporated grouping) of minimum two (2) and maximum
upto three (3) corporate entities and shall collectively meet the following qualification
requirements. """
result3 = qa.invoke(query)


clause_list = [ 
     {"details": "1.1.0 :"+ result1['result']},
      {"details": "1.2.0 :"+ result2['result']},
       {"details": "1.3.0 :"+ result3['result']}
]


# Load the template file
template = DocxTemplate("Evaluation_template.docx")

# Create a context dictionary with data to be inserted into the template
context = {
    'project_name': 'BALANCE OF PLANT (BOP) TURNKEY PACKAGE FOR RAGHUNATHPUR THERMAL POWER STATION PHASE-II (2X660MW)',
    'company_name': 'Damodar Valley Corporation',
    'project_address': 'DPURULIA DISTRICT OF WEST BENGAL',
    'project_code': 'DVC/C&M/Engineering/RTPS Ph-II/EPC/BOP',
    'client_name': 'Damodar Sharma',
    'start_date': date(2023, 1, 1).strftime("%d.%m.%Y"),
    'end_date': date(2023, 3, 31).strftime("%d.%m.%Y"),
    'tedate': datetime.now().strftime("%d.%m.%Y"),
    'submitter_name': 'Jane Smith',
    'clauses': clause_list 
    }

# Render the template with the context data
template.render(context)

# Save the generated document to a new file
template.save(f"SampleResponse_{context['project_name']}.docx")