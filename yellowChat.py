import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity       
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy



lemmatizer = nltk.stem.WordNetLemmatizer()


# Download required NLTK data ----- if you are running this program for the first time, you should uncomment the three lines of code after this
nltk.download('stopwords') # ----------- python -m nltk.downloader stopwords
nltk.download('punkt') # --------------- python -m nltk.downloader punkt
nltk.download('wordnet') # ------------- python -m nltk.downloader wordnet

df = pd.read_csv("Dataset\Mental_Health_FAQ.csv", na_filter=False)
df = df[['Questions', 'Answers']]


# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)

# Apply the function above 
df['tokenized Questions'] = df['Questions'].apply(preprocess_text)

# Create a corpus by flattening the preprocessed questions
corpus = df['tokenized Questions'].tolist()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)
# TDIDF is a numerical statistic used to evaluate how important a word is to a document in a collection or corpus. 
# The TfidfVectorizer calculates the Tfidf values for each word in the corpus and uses them to create a matrix where each row represents a document and each column represents a word. 
# The cell values in the matrix correspond to the importance of each word in each document.

def get_response(user_input):
    global most_similar_index
    
    user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

    user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

    similarity_scores = cosine_similarity(user_input_vector, X) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

    most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

    return df['Answers'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response

# create greeting list 
greetings = ["Hey There, How can I help",
            "Hi Human.... How can I help",
            "Good Day .... How can I help", 
            "Hello There... How can I be useful to you today",
            "Hi GomyCode Student.... How can I be of use"]

exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks....see you soon', 'Bye-bye, See you soon', 'Bye... See you later', 'Bye... come back soon']

random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list



# -------------------------- STREAMLIT IMPLEMENTATION ---------------------- 
# import streamlit as st
st.markdown("<h1 style = 'text-align: center; color: #176B87'>Mental Support ChatBot</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>Built Regusa</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('Dataset\pngwing.com (55).png', caption = 'Mental Health Related Chats')

history = []
st.sidebar.markdown("<h2 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>Chat History</h2>", unsafe_allow_html = True)

user_input = col2.text_input(f'Ask Your Question ')
if user_input:
    if user_input.lower() in exits:
        bot_reply = col2.write(f"\nChatbot\n: {random_farewell}!")
    if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
        bot_reply = col2.write(f"\nChatbot\n: {random_greetings}!")
    else:   
        response = get_response(user_input)
        bot_reply = col2.write(f"\nChatbot\n: {response}")
        
with open("chat_history.txt", "w") as file:
    file.write(user_input + "\n")

history.append(user_input)
# st.sidebar.write(history)
with open("chat_history.txt", "r") as file:
    history = file.readlines()

# st.text("Chat History:")

with st.sidebar:
    st.write('FAQs')
    st.write(df['Questions'][0:10])
    for message in history:
        st.write(message)