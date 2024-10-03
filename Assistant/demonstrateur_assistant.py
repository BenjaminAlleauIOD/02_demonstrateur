import streamlit as st
from openai import OpenAI
import os
import time
from PIL import Image

# Assurez-vous de définir la variable d'environnement OPENAI_API_KEY avant de lancer l'application.
token_api = st.secrets["token_api"]
client = OpenAI(api_key=token_api)

# Fonction pour attendre la fin de l'exécution
def wait_on_run(run, thread, client):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

def parse_response(messages):
    """Parse la réponse brute et extrait le texte pertinent."""
    t = str(messages.data[0].content[0].text).split("value=")[1]
    return t.replace('\\n', '\n').replace('"','').replace('")','')

# Récupération de l'ID de l'assistant stocké
assistant_id = "asst_KgGMLas5AxxmzZEnXuneEKBs"

# Interface Streamlit
st.set_page_config(layout="wide")

# Load image from file
img = Image.open("bot_iod.png")

# Créer deux colonnes
col1, col2 = st.columns([1, 5])  # Ajustez la largeur des colonnes si nécessaire
# Mettre l'image dans la première colonne
col1.image(img)  # Ajustez la largeur de l'image comme souhaité

# Mettre le titre dans la deuxième colonne
with col2:
    st.title("Bonjour, je suis Ioda!")
    st.title("Comment puis-je vous aider?")
st.subheader("Ioda est votre assistant personnalisé pour Socofer")

img_logo = Image.open("Logo_Iod_solutions_Horizontal_Logo_Complet_Blanc_RVB_1186px@72ppi.png")
st.sidebar.image(img_logo)
password = st.sidebar.text_input("Entrez le mot de passe", type="password")

if password == st.secrets["pwd"]:
    
    # Ajout de la possibilité de télécharger un document
    uploaded_files = st.file_uploader("Téléchargez un document", accept_multiple_files=True, type=["txt", "pdf", "docx"])
    
    # Variable pour stocker le contenu des fichiers
    file_content = ""
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Lire le contenu du fichier
            file_content += uploaded_file.getvalue().decode("utf-8")  # Supposant qu'il s'agit de fichiers texte
            st.write(f"Contenu du fichier {uploaded_file.name} chargé.")
    
    user_input = st.text_input('Posez votre question ici:')
    
    if st.button('Envoyer'):
        with st.spinner('Traitement en cours... Veuillez patienter.'):
            thread = client.beta.threads.create()
            
            # Inclure le contenu des fichiers dans la requête s'il y en a
            full_content = user_input + "\n\nContenu des fichiers : " + file_content if file_content else user_input
            
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=full_content,
            )

            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
            )

            run = wait_on_run(run, thread, client)

            messages = client.beta.threads.messages.list(thread_id=thread.id)
            parsed_response = parse_response(messages)

        st.markdown(parsed_response, unsafe_allow_html=True)  # Affichage de la réponse formatée
