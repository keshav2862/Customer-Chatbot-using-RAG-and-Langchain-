Steps to install the chatbot:
use python 3.9-3.11

1) Create a virtual environment

2) Install the required packages using pip install -r requirements.txt

3) Run the app.py file by using "python -m streamlit run app.py".

->The prompt can be edited accordingly in the App.py file.

->Upload the Robinhood document given in the docs folder.

->Additionally documents can be uploaded of your choice using the browse feature

->Use the "Chat with Persisted DB" option to chat with previously uploaded documents.

->Delete the vectorstore if you want to start afresh 

->IMPORTANT : Visual studio C++(MSVC Latest version) and Windows 10/11 SDK must be installed using the Visual Studio Installer for chromadb else use FAISS.

IF NO PROPER OPENAI KEY, DOWNLOAD AND USE MISTRAL-7b USING chatollama() function for local LLM

