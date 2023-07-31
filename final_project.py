#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess

# # Install Flask using pip
# subprocess.run(['pip', 'install', 'flask'])
# subprocess.run(['pip', 'install', 'tweet-preprocessor'])
# subprocess.run(['pip', 'install', 'textblob'])
# subprocess.run(['pip', 'install', 'sastrawi'])
# subprocess.run(['pip', 'install', 'emoji'])
# subprocess.run(['pip', 'install', 'PySastrawi'])
# subprocess.run(['pip', 'install', 'pandas'])
# subprocess.run(['pip', 'install', 'tweepy'])
# subprocess.run(['pip', 'install', 'seaborn'])
# subprocess.run(['pip', 'install', 'matplotlib'])
# subprocess.run(['pip', 'install', 'scikit-learn'])
# subprocess.run(['pip', 'install', 'wordcloud'])
# subprocess.run(['pip', 'install', 'apify-client'])
# subprocess.run(['pip', 'install', 'wordcloud'])
# subprocess.run(['pip', 'install', 'openpyxl'])
# subprocess.run(['pip', 'install', 'plotly'])

# Rest of your code...

# get_ipython().run_line_magic('pip', 'install tweet-preprocessor')
# get_ipython().run_line_magic('pip', 'install textblob')
# get_ipython().run_line_magic('pip', 'install sastrawi')
# get_ipython().run_line_magic('pip', 'install emoji')
# get_ipython().run_line_magic('pip', 'install PySastrawi')
# get_ipython().run_line_magic('pip', 'install pandas')
# get_ipython().run_line_magic('pip', 'install tweepy')
# get_ipython().run_line_magic('pip', 'install seaborn')
# get_ipython().run_line_magic('pip', 'install matplotlib')
# get_ipython().run_line_magic('pip', 'install scikit-learn')
# get_ipython().run_line_magic('pip', 'install wordcloud')
# get_ipython().run_line_magic('pip', 'install apify-client')
# get_ipython().run_line_magic('pip', 'install wordcloud')

import pandas as pd
import re
import string
from textblob import TextBlob
import string
import nltk
import preprocessor as p
from preprocessor.api import clean, tokenize, parse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import datetime
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from apify_client import ApifyClient
from flask import Flask, make_response, request, render_template, send_file, redirect
from IPython.display import display, HTML


app = Flask(__name__)

@app.route('/', methods=['GET'])
def login():
    html_content = """
        <!DOCTYPE html>
        <html lang="en">

        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <meta name="description" content="">
            <meta name="author" content="">
            <link rel="icon" href="/docs/4.0/assets/img/favicons/favicon.ico">

            <title>Login</title>

            <!-- Bootstrap core CSS -->
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css"
                integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

            <style>
                .divider:after,
                .divider:before {
                    content: "";
                    flex: 1;
                    height: 1px;
                    background: #eee;
                }

                .h-custom {
                    height: calc(100% - 73px);
                }

                @media (max-width: 450px) {
                    .h-custom {
                        height: 100%;
                    }
                }
            </style>
        </head>

        <body class="text-center">
            <section class="vh-100" style="height: 600px;">
                <div class="container-fluid h-custom">
                    <div class="row d-flex justify-content-center align-items-center h-100">
                        <div class="col-md-9 col-lg-6 col-xl-5">
                            <img
                                src="https://mdbcdn.b-cdn.net/img/Photos/new-templates/bootstrap-login-form/draw2.webp"
                                class="img-fluid" alt="Sample image">
                        </div>
                        <div class="col-md-8 col-lg-6 col-xl-4 offset-xl-1">
                            <form action="/dashboard" method="get" onsubmit="return validateForm()">
                                <div class="d-flex flex-row align-items-center justify-content-center justify-content-lg-start">
                                    <p class="lead fw-normal mb-0 me-3 font-weight-bold text-center">ANALISIS SENTIMEN
                                        INSTAGRAM</p>
                                </div>

                                <div class="divider d-flex align-items-center my-4">
                                    <p class="text-center fw-bold mx-3 mb-0"></p>
                                </div>

                                <!-- Username input with required attribute -->
                                <div class="form-outline mb-4">
                                    <input type="text" id="form3Example3" class="form-control form-control-lg"
                                        placeholder="Enter a username" required autocomplete="off"/>
                                    <input type="hidden" id="usernameInput" name="username" value="" />
                                    <!-- Custom validation message -->
                                    <div class="invalid-feedback">
                                        Wrong Username.
                                    </div>
                                </div>

                                <!-- Password input with required attribute -->
                                <div class="form-outline mb-3">
                                    <input type="password" id="form3Example4" class="form-control form-control-lg"
                                        placeholder="Enter password" required />
                                    <!-- Custom validation message -->
                                    <div class="invalid-feedback">
                                        Wrong Password.
                                    </div>
                                </div>

                                <div class="text-center text-lg-start mt-4 pt-2">
                                    <button type="submit" class="btn btn-dark btn-lg"
                                        style="padding-left: 2.5rem; padding-right: 2.5rem;">Login</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div
                    class="d-flex flex-column flex-md-row text-center text-md-start justify-content-between py-4 px-4 px-xl-5 bg-dark">
                    <!-- Copyright -->
                    <div class="text-white mb-3 mb-md-0">
                        Copyright © 2020. Nugroho - Analisis Sentimen.
                    </div>
                    <!-- Copyright -->
                </div>
            </section>

            <script>
                function validateForm() {
                    var usernameInput = document.getElementById('form3Example3');
                    var passwordInput = document.getElementById('form3Example4');

                    var username = usernameInput.value.trim();
                    var password = passwordInput.value.trim();

                    if (username === '' || (username !== 'admin' && username !== 'user')) {
                        usernameInput.classList.add('is-invalid');
                        usernameInput.classList.remove('is-valid');
                        usernameInput.setCustomValidity("Please enter a valid username.");
                        return false;
                    } else {
                        usernameInput.classList.remove('is-invalid');
                        usernameInput.classList.add('is-valid');
                        usernameInput.setCustomValidity("");
                    }

                    if (password === '' || (username === 'admin' && password !== 'password') || (username === 'user' && password !== 'password')) {
                        passwordInput.classList.add('is-invalid');
                        passwordInput.classList.remove('is-valid');
                        passwordInput.setCustomValidity("Please enter the correct password for the selected user.");
                        return false;
                    } else {
                        passwordInput.classList.remove('is-invalid');
                        passwordInput.classList.add('is-valid');
                        passwordInput.setCustomValidity("");
                    }
        
                    var usernameInputHidden = document.getElementById('usernameInput');
                    usernameInputHidden.value = username;
        
                    return true;
                }
            </script>
        </body>

        </html>

    """
    response = make_response(html_content)
    return response
  
@app.route('/dashboard', methods=['GET'])
def dashboard():
  username = request.args.get('username')
  print(username)
  
  if username == 'admin':
    import plotly.graph_objs as go
    # Load dataset
    dataset = pd.read_csv('labeling-data-instagram.csv')
    print(dataset)

    # Remove rows with NaN values
    dataset.dropna(subset=['clean_comment', 'label'], inplace=True)
    
    total_negative = dataset['label'].eq('negative').sum()
    total_neutral = dataset['label'].eq('neutral').sum()
    total_positive = dataset['label'].eq('positive').sum()

    return render_template('page-dashboard.html', total_positive=total_positive, total_negative=total_negative, total_neutral=total_neutral)
  else :
    # Render the test user dashboard
    return redirect('/user-prediction')
  
  

@app.route('/analisis-data/dataset', methods=['GET'])
def analisisDataset():
    csv_files = ['data-instagram.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('page-analisis-dataset.html', data=data)

@app.route('/crawling-data', methods=['GET'])
def pageCrawling():
    csv_files = ['data-instagram.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('page-crawling-data.html', data=data)

@app.route('/crawling-data', methods=['POST'])
def crawlingData():
    link = request.form.get('link')
    
    link = link.replace(" ","")
    link = link.split(",")
    
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_20esNplRw082naUn9ckCu6SXMIsWcE0YCS7h")

    # Prepare the actor input
    run_input = {
      "directUrls": link,
      "resultsLimit": 1500,
    }
    
    print(run_input)
    
    # Run the actor and wait for it to finish
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    db_comment = pd.DataFrame(columns=["userId","createdAt","text"])
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        value1 = item["id"]
        value2 = item["timestamp"]
        value3 = item["text"]
    
        # Create a new row with the values
        new_row = {"userId": value1, "createdAt": value2,"text": value3}
    
        # Append the new row to the DataFrame
        db_comment = pd.concat([db_comment, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    
    filename = "data-instagram.csv"
    db_comment.to_csv(filename, index=False) 
    
    csv_files = ['data-instagram.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('page-crawling-data.html', data=data)

@app.route('/analisis-data/preprocessing', methods=['GET'])
def preprocessing():
    #Read Data
    def load_data():
      data = pd.read_csv('data-instagram.csv')
      return data
    
    comment_df = load_data()
    comment_df = pd.DataFrame(comment_df[['userId', 'createdAt', 'text']])
    print(comment_df.head(1500))

    #cleaning
    def remove_pattern(text, pattern_regex):
      r = re.findall(pattern_regex, text)
      for i in r:
        text = re.sub(i, '', text)
        return text
      
    # remove tagging
    comment_df = comment_df[comment_df['text'].notnull()]
    comment_df['clean_tagging'] = np.vectorize(remove_pattern)(comment_df['text'], " *RT* | *@[\w]*")
    print(comment_df.head(10))
    
    #remove emoji & character
    def remove(text):
      text =' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split())
      return text
    comment_df['remove_character'] = comment_df['text'].apply(lambda x: remove(x))
    print(comment_df.head(9))
    
    #remove hastag
    def remov(text):
      text = re.sub(r'r\$\w*', '', text)
      text = re.sub(r'^RT[\s]+', '', text)
      text = re.sub(r'#', '', text)
      text = re.sub(r'[0-9]+', '',text)

      return text

    comment_df['remove_hastag'] = comment_df['remove_character'].apply(lambda x: remov(x))
    print(comment_df.head(10))
    
    #remove duplikat
    comment_df.drop_duplicates(subset = "remove_hastag", keep = 'first', inplace = True)
    print(comment_df.head(10))
    
    #import stopword
    nltk.download('stopwords')
    stopwords_indonesia = stopwords.words('indonesian')
    stopwords_indonesia

    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopwords = [
        'yg', 'utk', 'cuman', 'deh', 'Btw', 'tapi', 'gua', 'gue', 'lo', 'lu',
        'kalo', 'trs', 'jd', 'nih', 'ntr', 'nya', 'lg', 'gk', 'ecusli', 'dpt',
        'dr', 'kpn', 'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'km', 'eh',
        'sih', 'eh', 'bang', 'br', 'kyk', 'rp', 'jt', 'kan', 'gpp', 'sm', 'usah'
        'mas', 'sob', 'thx', 'ato', 'jg', 'gw', 'wkwkwk', 'mak', 'haha', 'iy', 'k'
        'tp','haha', 'dg', 'dri', 'duh', 'ye', 'wkwk', 'syg', 'btw',
        'nerjemahin', 'gaes', 'guys', 'moga', 'kmrn', 'nemu', 'yukk',
        'wkwkw', 'klas', 'iw', 'ew', 'lho', 'sbnry', 'org', 'gtu', 'bwt',
        'krlga', 'clau', 'lbh', 'cpet', 'ku', 'wke', 'mba', 'mas', 'sdh', 'kmrn',
        'oi', 'spt', 'dlm', 'bs', 'krn', 'jgn', 'sapa', 'spt', 'sh', 'wakakaka',
        'sihhh', 'hehe', 'ih', 'dgn', 'la', 'kl', 'ttg', 'mana', 'kmna', 'kmn',
        'tdk', 'tuh', 'dah', 'kek', 'ko', 'pls', 'bbrp', 'pd', 'mah', 'dhhh',
        'kpd', 'tuh', 'kzl', 'byar', 'si', 'sii', 'cm', 'sy', 'hahahaha', 'weh',
        'dlu', 'tuhh'
    ]
    data = stop_factory + more_stopwords

    dictionary = ArrayDictionary(data)
    stopWord = StopWordRemover(dictionary)

    print(data)

    #import Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    #tokenize
    from nltk.tokenize import TweetTokenizer

    #Happy Emoticon
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', ',x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*)', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-P', 'xp', ' XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3' 
    ])

    #Sad emoticon
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
      ':c', ':{', '>:\\', ';('
    ])

    #all emtoicons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)

    def clean_comment(comment):
      #tokenize
      tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
      comment_tokens = tokenizer.tokenize(comment)

      comments_clean = []
      for word in comment_tokens:
        if (
            word not in data and
            word not in emoticons and
            word not in string.punctuation):
            stem_word = stemmer.stem(word)
            comments_clean.append(stem_word)

      return comments_clean
    comment_df['tokenizing'] = comment_df ['remove_hastag'].apply(lambda x:clean_comment(x))
    #tokenization
    print(comment_df.head(10))
    
    #remove punct
    def remove_punct(text):
      text = " ".join([char for char in text if char not in string.punctuation])
      return text
    comment_df['clean_comment'] = comment_df ['tokenizing'].apply(lambda x:remove_punct(x))
    print(comment_df.head(10))
    
    #reset index
    comment_df = comment_df.reset_index(drop=True)
    print(comment_df.head(10))
    
    comment_df.drop_duplicates(subset ="remove_hastag", keep = 'first', inplace = True)
    print(comment_df.head(10))
    
    comment_df.to_csv('preprocessing-data.csv', encoding ='utf8', index = False)
    
    #remove kolom
    comment_df.drop(comment_df.columns[[0,1,3,4,5,6]], axis = 1, inplace = True)
    print(comment_df.head(1500))
    
    #simpan data bersih
    comment_df.to_csv('komentar_bersih_instagram.csv', encoding ='utf8', index = False)
    
    
    csv_files = ['preprocessing-data.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('page-analisis-preprocessing.html', data=data)

@app.route('/analisis-data/transformation', methods=['GET'])
def transformation():
    df = pd.read_csv('komentar_bersih_instagram.csv', usecols=['text','clean_comment']).astype('str')
    print(df.head(10))
    
    # LEXICON
    lexicon = pd.read_csv('lexicon.csv')
    lexicon['weight'] = lexicon['sentiment'].map({'positive':1, 'negative':-1}) 
    lexicon = dict(zip(lexicon['word'], lexicon['weight']))
    print(lexicon)
    
    # NEGATIVE WORDS
    negative_words = list(open("negative.txt"))
    negative_words = list([word.rstrip() for word in negative_words])
    print(negative_words)
    
    comment_polarity = [] 
    comment_weight = []
    negasi = False

    for sentence in df['clean_comment']: 
      sentence_score = 0 
      sentence_weight = "" 
      sentiment_count = 0 
      sentence = sentence.split()
      for word in sentence:
        try:
          score = lexicon[word]
          sentiment_count = sentiment_count + 1
        except:
          score = 99
    
        if(score == 99):
          if (word in negative_words): 
            negasi = True
            sentence_score = sentence_score - 1
            sentence_weight = sentence_weight + " - 1"
          else:
            sentence_score = sentence_score + 0 
            sentence_weight = sentence_weight + " + 0"
        else:
          if(negasi == True):
            sentence_score = sentence_score + (score * -1.0)
            sentence_weight = sentence_weight + " + ("+ str(score) + " * -1 "+") " 
            negasi = False
          else:
            sentence_score = sentence_score + score 
            sentence_weight = sentence_weight + " + "+ str(score)
        
      comment_weight.append(sentence_weight[1:] +" = " + str(sentence_score)) 
      if sentence_score > 0:
        comment_polarity.append('positive') 
      elif sentence_score < 0:
        comment_polarity.append('negative') 
      else:
        comment_polarity.append('neutral') 

    print(df.columns)
    results = pd.DataFrame({
        "original_comment" : df['text'], 
        "clean_comment" : df['clean_comment'], 
        "label" : comment_polarity, 
        "weight" : comment_weight
        })
    results['label'].value_counts()
    results[['original_comment','clean_comment', 'label', 'weight', ]].to_csv('labeling-data-instagram.csv', encoding ='utf8', index = False)

    df = pd.read_csv('labeling-data-instagram.csv', usecols=['clean_comment', 'label']).dropna() 

    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(df['clean_comment'])
    tfidf_df = pd.DataFrame(text_tf.toarray(), columns=tf.get_feature_names_out())

    # Simpan DataFrame ke dalam file CSV
    tfidf_df.to_csv('hasil_tfidf.csv', index=False)

    # Cetak hasil pembobotan
    print(tfidf_df)
    
    
    csv_files = ['labeling-data-instagram.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('page-analisis-transformation.html', data=data)

@app.route('/analisis-data/transformation/tf-idf', methods=['GET'])
def transformationTfidf():
    df = pd.read_csv('hasil_tfidf.csv')
    df.to_csv("hasil_tfidf.csv", index=False)
        
    return send_file("hasil_tfidf.csv", as_attachment=True)

@app.route('/analisis-data/datamining', methods=['GET'])
def datamining():
        
    return render_template('page-analisis-datamining.html')

@app.route('/analisis-data/datamining-process', methods=['GET'])
def dataminingProcess():
    presentase = request.args.get('selectedPresentase')
    persen = float(presentase)*100
    persen = int(persen)
  
    df = pd.read_csv('labeling-data-instagram.csv', usecols=['clean_comment', 'label']).dropna() 
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(df['clean_comment'])

    temporary_df = pd.DataFrame(text_tf.todense(), columns=tf.get_feature_names_out())
    temporary_df

    from sklearn.model_selection import train_test_split
    X_train,	X_test,	y_train,	y_test	=	train_test_split(text_tf,	df['label'],	test_size=float(presentase), random_state=42)
    
    from sklearn.naive_bayes import MultinomialNB
    from	sklearn.metrics	import	accuracy_score,	precision_score, recall_score, f1_score
    from	sklearn.metrics	import classification_report 
    from	sklearn.metrics	import confusion_matrix

    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    
    # For precision_score
    precision_positive = precision_score(y_test, predicted, labels=['positive'], average='macro')
    precision_negative = precision_score(y_test, predicted, labels=['negative'], average='macro')
    precision_neutral = precision_score(y_test, predicted, labels=['netral'], average='macro')

    # For recall_score
    recall_positive = recall_score(y_test, predicted, labels=['positive'], average='macro')
    recall_negative = recall_score(y_test, predicted, labels=['negative'], average='macro')
    recall_neutral = recall_score(y_test, predicted, labels=['netral'], average='macro')

    # Print the precision and recall for each class
    print("Multinomial NB Precision (Positive):", precision_positive)
    print("Multinomial NB Precision (Negative):", precision_negative)
    print("Multinomial NB Precision (Neutral):", precision_neutral)

    print("Multinomial NB Recall (Positive):", recall_positive)
    print("Multinomial NB Recall (Negative):", recall_negative)
    print("Multinomial NB Recall (Neutral):", recall_neutral)
    
    precision_positive = precision_score(y_test, predicted, labels=['positive'], average='macro', zero_division=1)
    precision_negative = precision_score(y_test, predicted, labels=['negative'], average='macro', zero_division=1)
    precision_neutral = precision_score(y_test, predicted, labels=['netral'], average='macro', zero_division=1)

    
    # print("Multinomial NB Accuracy  : ", accuracy_score(y_test,predicted))
    # print("Multinomial NB Precision : ", precision_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print("Multinomial NB Recall    : ", recall_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print("Multinomial NB F-Measure : ", f1_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print(classification_report(y_test, predicted, zero_division=0))    
    
    
    # Perform predictions and calculations
    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average='macro', pos_label="Positif")
    recall = recall_score(y_test, predicted, average='macro', pos_label="Positif")
    f_measure = f1_score(y_test, predicted, average='macro', pos_label="Positif")
    classification = classification_report(y_test, predicted, zero_division=0)
    
    
    import seaborn as sns 
    import matplotlib.pyplot as plt  
    original_comments = df.loc[y_test.index, 'clean_comment']
    results_df = pd.DataFrame({
        "Clean Comment": original_comments,
        "True Label": predicted,
        "Predicted Label": y_test
    })

    # Step 3: Print the DataFrame to see the results
    results_df.to_csv('compare-data.csv', encoding='utf-8', index=False)
    print(results_df)
    
    csv_files = ['compare-data.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('page-analisis-datamining-process.html', data=data,persen=persen,accuracy=accuracy, precision=precision, recall=recall, f_measure=f_measure, classification=classification)

@app.route('/analisis-data/evaluation', methods=['GET'])
def evaluation():
        
    return render_template('page-analisis-evaluation.html')

@app.route('/analisis-data/evaluation-process', methods=['GET'])
def evaluationProcess():
    presentase = request.args.get('selectedPresentase')
    persen = float(presentase)*100
    persen = int(persen)
  
    df = pd.read_csv('labeling-data-instagram.csv', usecols=['clean_comment', 'label']).dropna() 
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(df['clean_comment'])

    temporary_df = pd.DataFrame(text_tf.todense(), columns=tf.get_feature_names_out())
    temporary_df

    from sklearn.model_selection import train_test_split
    X_train,	X_test,	y_train,	y_test	=	train_test_split(text_tf,	df['label'],	test_size=float(presentase), random_state=42)
    
    from sklearn.naive_bayes import MultinomialNB
    from	sklearn.metrics	import	accuracy_score,	precision_score, recall_score, f1_score
    from	sklearn.metrics	import classification_report 
    from	sklearn.metrics	import confusion_matrix

    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    
    # # For precision_score
    # precision_positive = precision_score(y_test, predicted, labels=['positive'], average='macro')
    # precision_negative = precision_score(y_test, predicted, labels=['negative'], average='macro')
    # precision_neutral = precision_score(y_test, predicted, labels=['netral'], average='macro')

    # # For recall_score
    # recall_positive = recall_score(y_test, predicted, labels=['positive'], average='macro')
    # recall_negative = recall_score(y_test, predicted, labels=['negative'], average='macro')
    # recall_neutral = recall_score(y_test, predicted, labels=['netral'], average='macro')

    # # Print the precision and recall for each class
    # print("Multinomial NB Precision (Positive):", precision_positive)
    # print("Multinomial NB Precision (Negative):", precision_negative)
    # print("Multinomial NB Precision (Neutral):", precision_neutral)

    # print("Multinomial NB Recall (Positive):", recall_positive)
    # print("Multinomial NB Recall (Negative):", recall_negative)
    # print("Multinomial NB Recall (Neutral):", recall_neutral)
    
    # precision_positive = precision_score(y_test, predicted, labels=['positive'], average='macro', zero_division=1)
    # precision_negative = precision_score(y_test, predicted, labels=['negative'], average='macro', zero_division=1)
    # precision_neutral = precision_score(y_test, predicted, labels=['netral'], average='macro', zero_division=1)

    
    # print("Multinomial NB Accuracy  : ", accuracy_score(y_test,predicted))
    # print("Multinomial NB Precision : ", precision_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print("Multinomial NB Recall    : ", recall_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print("Multinomial NB F-Measure : ", f1_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print(classification_report(y_test, predicted, zero_division=0))    
    
    
    # Perform predictions and calculations
    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average='macro', pos_label="Positif")
    recall = recall_score(y_test, predicted, average='macro', pos_label="Positif")
    f_measure = f1_score(y_test, predicted, average='macro', pos_label="Positif")
    classification = classification_report(y_test, predicted, zero_division=0)
    
    import os
    import seaborn as sns 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    
    # os.remove(os.path.join("static", "confusion-matrix.png"))
    # os.remove(os.path.join("static", "grafic-roc.png"))
    # os.remove(os.path.join("static", "wrcld.png"))
    
    plt.close()
    cm = confusion_matrix(predicted, y_test) 
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix') 
    plt.savefig('static/confusion-matrix.png')
    # plt.show()
    plt.close()  
    
    predicted_proba = clf.predict_proba(X_test)
    predicted_proba
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    print(
        f'ROC	AUC	score:  {roc_auc_score(y_test, predicted_proba[:,	:], multi_class="ovr").round(4)}')
    
    
    y_test_bin = label_binarize(y_test, classes=['negative', 'neutral', 'positive']) 
    n_classes = y_test_bin.shape[1]
    fpr = dict() 
    tpr = dict()
    roc_auc = dict()
    roc_auc
    colors = ['violet', 'black', 'yellow'] 
    auc_scores = []
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predicted_proba[:, i]) 
      plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label= str(i)) 
      # print('AUC for Class {}: {}'.format(i, auc(fpr[i], tpr[i]).round(3)))
      # print()
      auc_score = auc(fpr[i], tpr[i]).round(3)
      auc_scores.append(auc_score)
      
    # plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves') 
    plt.legend(loc='lower right')
    plt.savefig('static/grafic-roc.png')
    # plt.show()
    plt.close()  


    isi_text = df['clean_comment']
    isi_text
    
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    #convert list to string and generate
    unique_string=(" ").join(df['clean_comment'])
    wordcloud = WordCloud(width = 800, height = 400,background_color ='white').generate(unique_string)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("static/wrcld.png", bbox_inches='tight')
    # # plt.show()
    plt.close()  
        
    return render_template('page-analisis-evaluation.html',persen=persen, accuracy=accuracy, precision=precision, recall=recall, f_measure=f_measure, classification=classification, auc_scores=auc_scores)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
  if request.method == 'POST':
    sentence = request.form['sentence']
    
    # Parsing kalimat
    parsed_sentence = parse_sentence(sentence)
    
    # Load dataset
    dataset = pd.read_csv('labeling-data-instagram.csv')
    
    # Remove rows with NaN values
    dataset.dropna(subset=['clean_comment', 'label'], inplace=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    # Split dataset into training and testing sets
    X = dataset['clean_comment']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform([parsed_sentence])


    # Train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vectorized, y_train)

    # Make prediction
    prediction = nb_classifier.predict(X_test_vectorized)[0]
    prediction_proba = nb_classifier.predict_proba(X_test_vectorized)[0]
    
    # Get the percentage for each sentiment category
    percentage_negative = prediction_proba[0] * 100
    percentage_neutral = prediction_proba[1] * 100
    percentage_positive = prediction_proba[2] * 100
    
    # Calculate accuracy on testing data
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = nb_classifier.score(X_test_vectorized, y_test)
    
    return render_template('page-prediction.html', sentence=sentence, 
            parsed_sentence=parsed_sentence, prediction=prediction,
            percentage_positive=percentage_positive,
            percentage_negative=percentage_negative,
            percentage_neutral=percentage_neutral,
            accuracy=accuracy)
  
  return render_template('page-prediction.html')

@app.route('/user-prediction', methods=['GET', 'POST'])
def userPrediction():
  if request.method == 'POST':
    sentence = request.form['sentence']
    
    # Parsing kalimat
    parsed_sentence = parse_sentence(sentence)
    
    # Load dataset
    dataset = pd.read_csv('labeling-data-instagram.csv')
    
    # Remove rows with NaN values
    dataset.dropna(subset=['clean_comment', 'label'], inplace=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    # Split dataset into training and testing sets
    X = dataset['clean_comment']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform([parsed_sentence])


    # Train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vectorized, y_train)

    # Make prediction
    prediction = nb_classifier.predict(X_test_vectorized)[0]
    prediction_proba = nb_classifier.predict_proba(X_test_vectorized)[0]
    
    # Get the percentage for each sentiment category
    percentage_negative = prediction_proba[0] * 100
    percentage_neutral = prediction_proba[1] * 100
    percentage_positive = prediction_proba[2] * 100
    
    # Calculate accuracy on testing data
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = nb_classifier.score(X_test_vectorized, y_test)
    
    return render_template('user-page-prediction.html', sentence=sentence, 
            parsed_sentence=parsed_sentence, prediction=prediction,
            percentage_positive=percentage_positive,
            percentage_negative=percentage_negative,
            percentage_neutral=percentage_neutral,
            accuracy=accuracy)
  
  return render_template('user-page-prediction.html')

def parse_sentence(sentence):
    
    # Pembersihan Tagging
    def remove_pattern(text, pattern_regex):
        r = re.findall(pattern_regex, text)
        for i in r:
            text = re.sub(i, '', text)
        return text

    sentence = remove_pattern(sentence, " *RT* | *@[\w]*")

    # Penghapusan Emoji dan Karakter Khusus
    def remove(text):
        text = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
        return text

    sentence = remove(sentence)

    # Penghapusan Hashtag
    def remov(text):
        text = re.sub(r'\$\w*', '', text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[0-9]+', '', text)
        return text

    sentence = remov(sentence)

    # Tokenisasi
    from nltk.tokenize import TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    comment_tokens = tokenizer.tokenize(sentence)

    # Stopword Removal
    nltk.download('stopwords')
    stopwords_indonesia = stopwords.words('indonesian')
    # Tambahkan stopwords tambahan jika diperlukan
    more_stopwords = []
    data = stopwords_indonesia + more_stopwords

    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopwords = [
        'yg', 'utk', 'cuman', 'deh', 'Btw', 'tapi', 'gua', 'gue', 'lo', 'lu',
        'kalo', 'trs', 'jd', 'nih', 'ntr', 'nya', 'lg', 'gk', 'ecusli', 'dpt',
        'dr', 'kpn', 'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'km', 'eh',
        'sih', 'eh', 'bang', 'br', 'kyk', 'rp', 'jt', 'kan', 'gpp', 'sm', 'usah'
        'mas', 'sob', 'thx', 'ato', 'jg', 'gw', 'wkwkwk', 'mak', 'haha', 'iy', 'k'
        'tp','haha', 'dg', 'dri', 'duh', 'ye', 'wkwk', 'syg', 'btw',
        'nerjemahin', 'gaes', 'guys', 'moga', 'kmrn', 'nemu', 'yukk',
        'wkwkw', 'klas', 'iw', 'ew', 'lho', 'sbnry', 'org', 'gtu', 'bwt',
        'krlga', 'clau', 'lbh', 'cpet', 'ku', 'wke', 'mba', 'mas', 'sdh', 'kmrn',
        'oi', 'spt', 'dlm', 'bs', 'krn', 'jgn', 'sapa', 'spt', 'sh', 'wakakaka',
        'sihhh', 'hehe', 'ih', 'dgn', 'la', 'kl', 'ttg', 'mana', 'kmna', 'kmn',
        'tdk', 'tuh', 'dah', 'kek', 'ko', 'pls', 'bbrp', 'pd', 'mah', 'dhhh',
        'kpd', 'tuh', 'kzl', 'byar', 'si', 'sii', 'cm', 'sy', 'hahahaha', 'weh',
        'dlu', 'tuhh'
    ]
    stop_factory = stop_factory+more_stopwords
    dictionary = ArrayDictionary(data)
    stopword = StopWordRemover(dictionary)

    # Proses Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Praproses komentar
    def clean_comment(comment):
        comments_clean = []
        for word in comment:
            if (
                word not in data and
                word not in string.punctuation
            ):
                stem_word = stemmer.stem(word)
                comments_clean.append(stem_word)

        return comments_clean

    sentence = clean_comment(comment_tokens)

    # Penggabungan kembali kata-kata yang telah dipreproses
    sentence = TreebankWordDetokenizer().detokenize(sentence)

    return sentence

@app.route('/metodologi', methods=['POST'])
def register():
    apifyToken = request.form.get('apifyToken')
    link = request.form.get('link')
    
    link = link.replace(" ","")
    link = link.split(",")
    
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_20esNplRw082naUn9ckCu6SXMIsWcE0YCS7h")

    # Prepare the actor input
    run_input = {
      "directUrls": [
        "https://www.instagram.com/p/CnN8rmyPtP5/",
        "https://www.instagram.com/p/CnNuuEyhirj/",
        "https://www.instagram.com/p/CoYoz-hOWnC/",
        "https://www.instagram.com/p/CnVstOXSy83/",
        "https://www.instagram.com/p/CngI0-LP5aW/",
        "https://www.instagram.com/p/CnOKSP4hAeO/",
        "https://www.instagram.com/p/CnOJIlBpzLm/",
        "https://www.instagram.com/p/CocOljmP5Yr/",
        "https://www.instagram.com/p/Cnf78BPrlAY/",
        "https://www.instagram.com/p/CnjdbCcvQAu/",
        "https://www.instagram.com/p/CnWigBDLDJj/",
        "https://www.instagram.com/p/CnPJnVKPA3A/",
        "https://www.instagram.com/p/CnRk109JpsU/",
        "https://www.instagram.com/p/CnOJcN1PryS/",
        "https://www.instagram.com/p/CobTZF8v17L/",
        "https://www.instagram.com/p/CnOOkZoJTUN/",
        "https://www.instagram.com/p/CnPaBUuSVax/",
        "https://www.instagram.com/p/CnQlUtOhXcr/",
        "https://www.instagram.com/p/CnOF-wFpM6V/",
        "https://www.instagram.com/p/CoKjvs5Lttk/",
        "https://www.instagram.com/p/CnQ5g0npO0I/",
        "https://www.instagram.com/p/Cnjg7DAB-Te/",
        "https://www.instagram.com/p/CnLS_vmpq34/",
        "https://www.instagram.com/p/CnL2HC9Scsh/",
        "https://www.instagram.com/p/CnQ_tNGvapq/",
        "https://www.instagram.com/p/B4_MvZGHD-h/",
        "https://www.instagram.com/p/CngEw2TrH0u/",
        "https://www.instagram.com/p/B5Fqc7vFHTA/",
        "https://www.instagram.com/p/Cniiz-WrbQ-/",
        "https://www.instagram.com/p/B5WhligFJ1D/",
        "https://www.instagram.com/p/CoeUl0xSbE1/",
    ],
      "resultsLimit": 1500,
    }
    
    print(run_input)
    
    # Run the actor and wait for it to finish
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    db_comment = pd.DataFrame(columns=["userId","createdAt","text"])
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        value1 = item["id"]
        value2 = item["timestamp"]
        value3 = item["text"]
    
        # Create a new row with the values
        new_row = {"userId": value1, "createdAt": value2,"text": value3}
    
        # Append the new row to the DataFrame
        db_comment = pd.concat([db_comment, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    
    filename = "data-instagram.csv"
    db_comment.to_csv(filename, index=False)
    
    # ######################################################################################
    # ################################ PREPROCESSING DATA ##################################
    # ######################################################################################
    
    #Read Data
    def load_data():
      data = pd.read_csv('data-instagram.csv')
      return data
    
    comment_df = load_data()
    comment_df = pd.DataFrame(comment_df[['userId', 'createdAt', 'text']])
    print(comment_df.head(1500))

    #cleaning
    def remove_pattern(text, pattern_regex):
      r = re.findall(pattern_regex, text)
      for i in r:
        text = re.sub(i, '', text)
        return text
      
    # remove tagging
    comment_df = comment_df[comment_df['text'].notnull()]
    comment_df['clean_tagging'] = np.vectorize(remove_pattern)(comment_df['text'], " *RT* | *@[\w]*")
    print(comment_df.head(10))
    
    #remove emoji & character
    def remove(text):
      text =' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split())
      return text
    comment_df['remove_character'] = comment_df['text'].apply(lambda x: remove(x))
    print(comment_df.head(9))
    
    #remove hastag
    def remov(text):
      text = re.sub(r'r\$\w*', '', text)
      text = re.sub(r'^RT[\s]+', '', text)
      text = re.sub(r'#', '', text)
      text = re.sub(r'[0-9]+', '',text)

      return text

    comment_df['remove_hastag'] = comment_df['remove_character'].apply(lambda x: remov(x))
    print(comment_df.head(10))
    
    #remove duplikat
    comment_df.drop_duplicates(subset = "remove_hastag", keep = 'first', inplace = True)
    print(comment_df.head(10))
    
    #import stopword
    nltk.download('stopwords')
    stopwords_indonesia = stopwords.words('indonesian')
    stopwords_indonesia

    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopwords = [
        'yg', 'utk', 'cuman', 'deh', 'Btw', 'tapi', 'gua', 'gue', 'lo', 'lu',
        'kalo', 'trs', 'jd', 'nih', 'ntr', 'nya', 'lg', 'gk', 'ecusli', 'dpt',
        'dr', 'kpn', 'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'km', 'eh',
        'sih', 'eh', 'bang', 'br', 'kyk', 'rp', 'jt', 'kan', 'gpp', 'sm', 'usah'
        'mas', 'sob', 'thx', 'ato', 'jg', 'gw', 'wkwkwk', 'mak', 'haha', 'iy', 'k'
        'tp','haha', 'dg', 'dri', 'duh', 'ye', 'wkwk', 'syg', 'btw',
        'nerjemahin', 'gaes', 'guys', 'moga', 'kmrn', 'nemu', 'yukk',
        'wkwkw', 'klas', 'iw', 'ew', 'lho', 'sbnry', 'org', 'gtu', 'bwt',
        'krlga', 'clau', 'lbh', 'cpet', 'ku', 'wke', 'mba', 'mas', 'sdh', 'kmrn',
        'oi', 'spt', 'dlm', 'bs', 'krn', 'jgn', 'sapa', 'spt', 'sh', 'wakakaka',
        'sihhh', 'hehe', 'ih', 'dgn', 'la', 'kl', 'ttg', 'mana', 'kmna', 'kmn',
        'tdk', 'tuh', 'dah', 'kek', 'ko', 'pls', 'bbrp', 'pd', 'mah', 'dhhh',
        'kpd', 'tuh', 'kzl', 'byar', 'si', 'sii', 'cm', 'sy', 'hahahaha', 'weh',
        'dlu', 'tuhh'
    ]
    data = stop_factory + more_stopwords

    dictionary = ArrayDictionary(data)
    stopWord = StopWordRemover(dictionary)

    print(data)

    #import Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    #tokenize
    from nltk.tokenize import TweetTokenizer

    #Happy Emoticon
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', ',x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*)', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-P', 'xp', ' XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3' 
    ])

    #Sad emoticon
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
      ':c', ':{', '>:\\', ';('
    ])

    #all emtoicons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)

    def clean_comment(comment):
      #tokenize
      tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
      comment_tokens = tokenizer.tokenize(comment)

      comments_clean = []
      for word in comment_tokens:
        if (
            word not in data and
            word not in emoticons and
            word not in string.punctuation):
            stem_word = stemmer.stem(word)
            comments_clean.append(stem_word)

      return comments_clean
    comment_df['clean_comment'] = comment_df ['remove_hastag'].apply(lambda x:clean_comment(x))
    #tokenization
    print(comment_df.head(10))
    
    #remove punct
    def remove_punct(text):
      text = " ".join([char for char in text if char not in string.punctuation])
      return text
    comment_df['clean_comment'] = comment_df ['clean_comment'].apply(lambda x:remove_punct(x))
    print(comment_df.head(10))
    
    #reset index
    comment_df = comment_df.reset_index(drop=True)
    print(comment_df.head(10))
    
    comment_df.drop_duplicates(subset ="remove_hastag", keep = 'first', inplace = True)
    print(comment_df.head(10))
    
    #remove kolom
    comment_df.drop(comment_df.columns[[0,1,2,3,4,5]], axis = 1, inplace = True)
    print(comment_df.head(1500))
    
    #simpan data bersih
    comment_df.to_csv('komentar_bersih_instagram.csv', encoding ='utf8', index = False)
    
    
    
    print("######################################################################################")
    print("################################## WEIGHT SENTIMENT ##################################")
    print("######################################################################################")
    
    df = pd.read_csv('komentar_bersih_instagram.csv', usecols=['clean_comment']).astype('str')
    print(df.head(10))
    
    # LEXICON
    lexicon = pd.read_csv('lexicon.csv')
    lexicon['weight'] = lexicon['sentiment'].map({'positive':1, 'negative':-1}) 
    lexicon = dict(zip(lexicon['word'], lexicon['weight']))
    print(lexicon)
    
    # NEGATIVE WORDS
    negative_words = list(open("negative.txt"))
    negative_words = list([word.rstrip() for word in negative_words])
    print(negative_words)
    
    comment_polarity = [] 
    comment_weight = []
    negasi = False

    for sentence in df['clean_comment']: 
      sentence_score = 0 
      sentence_weight = "" 
      sentiment_count = 0 
      sentence = sentence.split()
      for word in sentence:
        try:
          score = lexicon[word]
          sentiment_count = sentiment_count + 1
        except:
          score = 99
    
        if(score == 99):
          if (word in negative_words): 
            negasi = True
            sentence_score = sentence_score - 1
            sentence_weight = sentence_weight + " - 1"
          else:
            sentence_score = sentence_score + 0 
            sentence_weight = sentence_weight + " + 0"
        else:
          if(negasi == True):
            sentence_score = sentence_score + (score * -1.0)
            sentence_weight = sentence_weight + " + ("+ str(score) + " * -1 "+") " 
            negasi = False
          else:
            sentence_score = sentence_score + score 
            sentence_weight = sentence_weight + " + "+ str(score)
        
      comment_weight.append(sentence_weight[1:] +" = " + str(sentence_score)) 
      if sentence_score > 0:
        comment_polarity.append('positive') 
      elif sentence_score < 0:
        comment_polarity.append('negative') 
      else:
        comment_polarity.append('neutral') 

    results = pd.DataFrame({
        "comment" : df['clean_comment'], 
        "label" : comment_polarity, 
        "weight" : comment_weight
        })
    results['label'].value_counts()
    results[['comment', 'label']].to_csv('labeling-data-instagram.csv', encoding ='utf8', index = False)
    print(results.head(20))
    
    
    print("######################################################################################")
    print("#################################### DATA MINING #####################################")
    print("######################################################################################")

    df = pd.read_csv('labeling-data-instagram.csv', usecols=['comment', 'label']).dropna() 

    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(df['comment'])

    temporary_df = pd.DataFrame(text_tf.todense(), columns=tf.get_feature_names_out())
    temporary_df

    from sklearn.model_selection import train_test_split
    X_train,	X_test,	y_train,	y_test	=	train_test_split(text_tf,	df['label'],	test_size=0.1, random_state=42)
    
    from sklearn.naive_bayes import MultinomialNB
    from	sklearn.metrics	import	accuracy_score,	precision_score, recall_score, f1_score
    from	sklearn.metrics	import classification_report 
    from	sklearn.metrics	import confusion_matrix

    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    # print("Multinomial NB Accuracy  : ", accuracy_score(y_test,predicted))
    # print("Multinomial NB Precision : ", precision_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print("Multinomial NB Recall    : ", recall_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print("Multinomial NB F-Measure : ", f1_score(y_test,predicted, average = 'macro', pos_label="Positif"))
    # print(classification_report(y_test, predicted, zero_division=0))    
    
    
    # Perform predictions and calculations
    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average='macro', pos_label="Positif")
    recall = recall_score(y_test, predicted, average='macro', pos_label="Positif")
    f_measure = f1_score(y_test, predicted, average='macro', pos_label="Positif")
    classification = classification_report(y_test, predicted, zero_division=0)
    
    
    import seaborn as sns 
    import matplotlib.pyplot as plt 
    cm = confusion_matrix(predicted, y_test) 
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix') 
    plt.savefig('static/confusion-matrix.png')
    # plt.show()
    # plt.close()  
    
    predicted_proba = clf.predict_proba(X_test)
    predicted_proba
    
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    print(
        f'ROC	AUC	score:  {roc_auc_score(y_test, predicted_proba[:,	:], multi_class="ovr").round(4)}')
    
    y_test_bin = label_binarize(y_test, classes=['negative', 'neutral', 'positive']) 
    n_classes = y_test_bin.shape[1]
    fpr = dict() 
    tpr = dict()
    roc_auc = dict()
    roc_auc
    colors = ['violet', 'black', 'yellow'] 
    auc_scores = []
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predicted_proba[:, i]) 
      plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label= str(i)) 
      # print('AUC for Class {}: {}'.format(i, auc(fpr[i], tpr[i]).round(3)))
      # print()
      auc_score = auc(fpr[i], tpr[i]).round(3)
      auc_scores.append(auc_score)
      
    # plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves') 
    plt.legend(loc='lower right')
    plt.savefig('static/grafic-roc.png')
    # plt.show()
    # plt.close()  


    isi_text = df['comment']
    isi_text
    
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    #convert list to string and generate
    unique_string=(" ").join(df['comment'])
    wordcloud = WordCloud(width = 800, height = 400,background_color ='white').generate(unique_string)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("static/wrcld.png", bbox_inches='tight')
    # plt.show()
    # plt.close()  
    
    
    csv_files = ['data-instagram.csv', 'komentar_bersih_instagram.csv', 'labeling-data-instagram.csv']
    data = []
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        df = df.head(10)
        # Convert the DataFrame to a dictionary and append it to the data list
        data.append(df.to_dict('records'))
        
    return render_template('index.html', data=data,accuracy=accuracy, precision=precision, recall=recall, f_measure=f_measure, classification=classification, auc_scores=auc_scores)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
    # app.run(debug=True, host='0.0.0.0', port=5050)

