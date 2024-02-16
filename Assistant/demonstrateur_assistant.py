import streamlit as st
from openai import OpenAI
import os
import time
from PIL import Image


# Assurez-vous de définir la variable d'environnement OPENAI_API_KEY avant de lancer l'application.
token_api = os.environ["token_api"]

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
# Assurez-vous de stocker l'assistant_id dans un endroit sûr et de le charger d'une manière qui ne compromet pas la sécurité.
assistant_id = "asst_gazNUtNLURVTvzZoSa5jtafS"

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
# st.image(img)
# st.title("Bonjour, je suis Ioda! comment puis-je vous aider?")
st.subheader("Ioda est spécialisé dans les réglementations des contenants alimentaires")

img_logo = Image.open("Logo_Iod_solutions_Horizontal_Logo_Complet_Original_RVB_1186px@72ppi (1).png")
st.sidebar.image(img_logo)
password = st.sidebar.text_input("Entrez le mot de passe", type="password")

if password == os.environ["pwd"]:
    user_input = st.text_input('Posez votre question ici:')

    if st.button('Envoyer'):
        with st.spinner('Traitement en cours... Veuillez patienter.'):
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_input,
            )

            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
            )

            run = wait_on_run(run, thread, client)

            messages = client.beta.threads.messages.list(thread_id=thread.id)
            parsed_response = parse_response(messages)

        st.markdown(parsed_response,unsafe_allow_html=True)  # Affichage de la réponse formatée
