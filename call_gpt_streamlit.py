import streamlit as st
from openai import OpenAI
import os
import time

# Fonction pour attendre la fin de l'exécution
def wait_on_run(run, thread, client):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

# Remplacez par votre token API personnel
token_api = "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"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", token_api))

# Création de l'assistant réglementaire
assistant = client.beta.assistants.create(
    name="Assistant réglementaire",
    instructions="tu seras spécialisé dans les reglementations des contenants alimentaire en français. Renvoi des réponses sous forme de bullet point. donne les articles spécifiant les normes CE, NF",
    model="gpt-4",
)

# Interface Streamlit
st.title('Assistant Réglementaire pour Contenants Alimentaires')

user_input = st.text_input('Posez votre question ici:')

if st.button('Envoyer'):
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    run = wait_on_run(run, thread, client)

    messages = client.beta.threads.messages.list(thread_id=thread.id)

    for message in messages.data:
        if message.role == "assistant":
            st.write(message.content)
