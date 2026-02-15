import sys
import os
import requests
import pandas as pd
import numpy as np
from io import StringIO

# Here I am making the importation of the visualization libraries.
# These are required for the plotting of the graphs and charts.
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# The sklearn modules are imported here by me for the text processing.
# These are very essential for the conversion of text to numbers.
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# In this section I am importing all the 10 classifiers which are needed for comparison.
# All the different models are loaded from the sklearn library.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# --- part 1 : getting data ---

def load_bbc_dat():
    """
    This function is written for the purpose of loading the dataset.
    It fetches the data from the internet directly.
    """
    
    # The url variable is defined here which holds the link to the csv file.
    my_url = "https://raw.githubusercontent.com/mdsohaib/BBC-News-Classification/master/bbc-text.csv"
    print(f"getting the data from : {my_url} ...")

    try:
        # The request is sent to the url to get the content of the file.
        r = requests.get(my_url)
        
        # Here the reading of the csv content is done using pandas.
        # StringIO is utilized to treat the text string as a file object.
        df_raw = pd.read_csv(StringIO(r.text))
        
        # I am filtering the dataframe here.
        # Only the categories of 'sport' and 'politics' are needed by us.
        cats_needed = ['sport', 'politics']
        # The copy method is used to create a separate copy of the filtered data.
        df_final = df_raw[df_raw['category'].isin(cats_needed)].copy()
        
        # The mapping of the categorical labels is performed here.
        # Politics is assigned the value 0 and Sport is assigned the value 1.
        df_final['cat_id'] = df_final['category'].map({'politics': 0, 'sport': 1})
        
        # Printing the details of the loaded data to the console.
        print(f"done loading. total rows : {len(df_final)}")
        print(f"sport count: {sum(df_final['cat_id']==1)} | politics count: {sum(df_final['cat_id']==0)}\n")

        # The function is returning the text list and the label list.
        return df_final['text'].tolist(), df_final['cat_id'].tolist()
        
    except Exception as e:
        # If any error occurs during the download process it is caught here.
        print(f"error in download : {e}")
        sys.exit(1)


# --- part 2 : eda visuals ---

def do_eda_tasks(txt_data, y_vals):
    """
    The execution of Exploratory Data Analysis tasks happens in this function.
    Various plots are generated to understand the data structure.
    """
    print("... running eda part ...")
    
    # This is the first plot. It is for checking the balance of the classes.
    # It helps to know if one class is having more samples than the other.
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_vals, hue=y_vals, palette=['#FF9999', '#66B2FF'], legend=False)
    plt.xticks([0, 1], ['Politics', 'Sports'])
    plt.title('class balance check')
    plt.savefig('fig1_eda_distribution.png')


    # Here I am calculating the length of each article in terms of words.
    # A list comprehension is used for this purpose.
    lns = [len(x.split()) for x in txt_data]
    
    # The histogram is plotted here to show the distribution of lengths.
    plt.figure(figsize=(8, 5))
    sns.histplot(lns, kde=True, color='teal')
    plt.title('article length distribution')
    plt.xlabel('words count')
    plt.savefig('fig2_eda_lengths.png')


    # This is a helper function defined inside.
    # It is used for the plotting of n-grams which are phrases of words.
    def plt_ngrams(lst_txt, ti_tle, f_name, col):
        # CountVectorizer is initialized with specific parameters.
        # The ngram_range is set to (2,2) to get only bi-grams.
        cv = CountVectorizer(stop_words='english', ngram_range=(2,2), max_features=10)
        
        # Fitting the vectorizer to the text list.
        mat = cv.fit_transform(lst_txt)
        # Summing up the counts of each bi-gram.
        sum_val = mat.sum(axis=0)
        
        # The sorting of the frequencies is done here in descending order.
        frq = sorted([(w, sum_val[0, i]) for w, i in cv.vocabulary_.items()], key=lambda x: x[1], reverse=True)
        
        # Creating a temporary dataframe for easy plotting with seaborn.
        df_tmp = pd.DataFrame(frq, columns=['phrase', 'cnt'])
        
        # The bar plot is created here using the dataframe.
        plt.figure(figsize=(8, 5))
        sns.barplot(x='cnt', y='phrase', data=df_tmp, color=col)
        plt.title(ti_tle)
        plt.tight_layout()
        plt.savefig(f_name)


    # Here I am separating the text data based on the labels.
    # One list is for politics and one list is for sports.
    p_txt = [txt_data[i] for i, l in enumerate(y_vals) if l == 0]
    s_txt = [txt_data[i] for i, l in enumerate(y_vals) if l == 1]
    
    # Calling the helper function to plot bi-grams for both categories.
    plt_ngrams(p_txt, 'top politics phrases', 'fig3_eda_pol_phrases.png', '#e74c3c')
    plt_ngrams(s_txt, 'top sports phrases', 'fig4_eda_spt_phrases.png', '#3498db')


    # This section is for the generation of Word Clouds.
    # It shows the most frequent words in a graphical manner.
    wc1 = WordCloud(width=600, height=300, background_color='white').generate(" ".join(p_txt))
    wc2 = WordCloud(width=600, height=300, background_color='white').generate(" ".join(s_txt))
    
    # Subplots are used to show both word clouds side by side.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].imshow(wc1)
    ax[0].set_title('politics vocab')
    ax[0].axis('off')
    
    ax[1].imshow(wc2)
    ax[1].set_title('sports vocab')
    ax[1].axis('off')
    
    plt.savefig('fig5_wordclouds.png')
    
    print("eda images saved.\n")


# --- part 3 : model training ---

def build_and_test(txt_data, y_vals):
    """
    This function contains the main logic for model building.
    It trains multiple models and evaluates their performance.
    """
    print("... starting training ...")
    
    # The TF-IDF vectorizer is initialized here.
    # It is used to convert the raw text into numerical features.
    # Max features is limited to 3000 to avoid high dimensionality issues.
    vec_tf = TfidfVectorizer(stop_words='english', max_features=3000)
    x_mat = vec_tf.fit_transform(txt_data)
    
    # Splitting the data into training set and testing set.
    # The test size is kept at 20 percent of the total data.
    x_tr, x_te, y_tr, y_te = train_test_split(x_mat, y_vals, test_size=0.2, random_state=42)
    
    # A dictionary is defined here containing all the 10 models.
    # Each model is initialized with specific parameters.
    clfs = {
        "1. Naive Bayes": MultinomialNB(),
        "2. Log Regression": LogisticRegression(max_iter=1000),
        "3. Rand Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "4. SVM (Linear)": SVC(kernel='linear', random_state=42),
        "5. KNN": KNeighborsClassifier(n_neighbors=9),
        "6. Grad Boosting": GradientBoostingClassifier(random_state=42),
        "7. Dec Tree": DecisionTreeClassifier(random_state=42),
        "8. Neural Net": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "9. Nearest Centroid": NearestCentroid(),
        "10. AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42)
    }
    
    nms = []
    scrs = []
    bst_sc = 0
    winner_mdl = None

    # Iterating through the dictionary items to train each model.
    for nm, mdl in clfs.items():
        # The model is fitted on the training data here.
        mdl.fit(x_tr, y_tr)
        
        # Prediction is made on the test data.
        pred = mdl.predict(x_te)
        # Accuracy score is calculated by comparing prediction with actual values.
        curr_acc = accuracy_score(y_te, pred)
        
        print(f"{nm:<20} | accuracy : {curr_acc:.4f}")
        
        nms.append(nm)
        scrs.append(curr_acc)
        
        # Checking if current model is better than the best score so far.
        if curr_acc > bst_sc:
            bst_sc = curr_acc
            winner_mdl = mdl
            
    
    # --- results visuals ---
    
    print("\n... generating result plots ...")
    
    # Plotting the comparison of accuracies of different models.
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scrs, y=nms, hue=nms, palette='magma', legend=False)
    plt.xlim(0.90, 1.01)
    plt.title('model comparison')
    plt.tight_layout()
    plt.savefig('fig6_accuracy_comparison.png')

    # t-SNE visualization is performed here.
    # It is a technique for dimensionality reduction to verify separability.
    print("running t-sne ...")
    
    # TruncatedSVD is used first to reduce dimensions to 50.
    # This is recommended before using t-SNE on sparse data.
    svd_red = TruncatedSVD(n_components=50, random_state=42).fit_transform(x_mat)
    # Applying t-SNE to reduce to 2 dimensions for plotting.
    tsne_res = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(svd_red)
    
    # Scatter plot is created to visualize the clusters.
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=y_vals, cmap='coolwarm', alpha=0.5)
    plt.title('t-sne clusters')
    plt.savefig('fig7_tsne_clusters.png')
    
    print("all plots saved.")
    
    # Returning the best model and the vectorizer object.
    return winner_mdl, vec_tf


def run_pred(f_path, mdl, v_ec):
    """
    This function is responsible for prediction on a new file.
    It takes the file path and model as input.
    """
    
    # Checking if the file path exists or not.
    if not os.path.exists(f_path):
        print(f"file not found : {f_path}")
        return

    # Reading the content of the file.
    with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Replacing newlines with spaces for better processing.
        raw_t = f.read().replace('\n', ' ')
        
    # The new text is transformed using the already fitted vectorizer.
    vec_in = v_ec.transform([raw_t])
    # Making the prediction using the best model.
    res_val = mdl.predict(vec_in)[0]
    
    # Converting the numerical result back to string label.
    out_str = 'SPORTS' if res_val == 1 else 'POLITICS'
    
    print(f"\nfile analyzed: {f_path}")
    print(f"final prediction: {out_str}")


if __name__ == "__main__":
    
    # Checking if the user has provided the file path argument.
    if len(sys.argv) < 2:
        print("usage: python B23CM1051_prob4.py <test_file.txt>")
        sys.exit(1)
        
    arg_f = sys.argv[1]
    
    # 1. Calling the function to load the data.
    docs_list, y_list = load_bbc_dat()
    
    # 2. Executing the EDA tasks.
    do_eda_tasks(docs_list, y_list)
    
    # 3. Training the models and getting the best one.
    best_m, tf_v = build_and_test(docs_list, y_list)
    
    # 4. Running the prediction on the provided test file.
    run_pred(arg_f, best_m, tf_v)

