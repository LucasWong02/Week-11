import openai
import gradio as gr
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import requests
from PIL import Image
from io import BytesIO

CONTEXT_TOKEN_LIMIT = 500
openai.api_key = "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"
# Initialize the variable to hold the processed file data
processed_data = None

def process_file(file):
    global processed_data
    if file.name.endswith(".pdf"):
        #Importar PDF
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()
        split = CharacterTextSplitter(chunk_size=200, separator='.\n')
        textos = split.split_documents(pages)
        print(textos)
        textos = [str(i.page_content).replace(")", "").replace("(", "").replace("S/000", "") for i in textos]
        parrafos = pd.DataFrame(textos, columns=["texto"])
        parrafos['Embedding'] = parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        processed_data = parrafos
        texto = str(processed_data.loc[37][0])
        #print(texto)
        #Importar CSV
        #loader = CSVLoader("/Users/lucaswongmang/Downloads/_Financial statements related procedures-Estatutaria.csv")
        # Create an index using the loaded documents
        #index_creator = VectorstoreIndexCreator()
        #docsearch = index_creator.from_loaders([loader])
        #print(docsearch)
        #chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
        #3 retriever = docsearch.vectorstore.as_retriever(), input_key="question")
        #frames = [processed_data, chain]
        #result = pd.concat(frames)
        #print(result)
    elif file.name.endswith(".csv"):
        processed_data = pd.read_csv(file.name)
    elif file.name.endswith(".doc"):
        # Process .doc file
        pass  # Add your code to process .doc file here

def buscar(busqueda, datos, n_resultados=5):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["texto", "Similitud", "Embedding"]]

def askPrueba(question: str):
    global processed_data
    if processed_data is None:
        print("No file has been uploaded yet.")
        return
    ordered_candidates = buscar(question, processed_data)
    sources = ordered_candidates.loc[:, "texto"]
    ctx = ""
    for candi in sources:
        next = ctx + " " + str(candi)
        ctx = next
    if len(ctx) == 0:
        print("")
    prompt = "Respondeme la pregunta en base al contexto:\n" + "contexto:" + str(ctx) + u"\n\n" + "Q:" + question
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    with open('log.txt', 'w') as f:
        f.write(completion.choices[0].message.content.replace(",000", ""))
    return completion.choices[0].message.content.replace(",000", "")
def user_help(do):
    return "So today we will do " + do + "using Gradio. Great choice!"

def upload_file(files):
    process_file(files)

def get_image(user_prompt):
    openai.api_key = "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"
    response = openai.Image.create (
        prompt=user_prompt,
        n=1,
        size="1024x1024"
    )
    url1 = str(response['data'][0]['url'])
    r = requests.get(url1)
    i = Image.open(BytesIO(r.content))
    return i

#app2 =  gr.Interface(fn = user_help, inputs="text", outputs="text")
app2 = gr.Interface(fn=get_image, inputs = [gr.Textbox(label="Enter the Prompt")], outputs = gr.Image(type='pil'))

with gr.Blocks() as app1:
    busqueda = gr.Textbox(label="Buscar")
    output = gr.Textbox(label="Respuesta")
    greet_btn = gr.Button("Preguntar")
    upload_button = gr.UploadButton(file_types=[".pdf", ".csv", ".doc"])
    upload_button.upload(upload_file, upload_button)
    greet_btn.click(fn=askPrueba, inputs=[busqueda], outputs=output)

demo = gr.TabbedInterface([app1, app2], ["Chat-PDF", "Imagine Image"])


demo.launch()
