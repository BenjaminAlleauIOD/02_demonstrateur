import streamlit as st
from openai import OpenAI
import os
import time

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
    return t.replace('\\n', '\n')

# Récupération de l'ID de l'assistant stocké
# Assurez-vous de stocker l'assistant_id dans un endroit sûr et de le charger d'une manière qui ne compromet pas la sécurité.
assistant_id = "asst_gazNUtNLURVTvzZoSa5jtafS"

# Interface Streamlit
st.title('Assistant Réglementaire pour Contenants Alimentaires')

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
