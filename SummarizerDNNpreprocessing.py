import pandas as pd
from nltk.corpus import re
from nltk.corpus import stopwords
import pickle
import nltk
nltk.download('stopwords')


reviews = pd.read_csv("Reviews.csv")
print(reviews.shape)
print(reviews.head())
print(reviews.isnull().sum())


reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator', 'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)
print(reviews.head())
for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()


contractions = {"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have"}


def clean_text(text, remove_stopwords=True):# Convert words to lower case
    text = text.lower()
    #if True:text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text


# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")
clean_texts = []
# for text in reviews.Text:
#     clean_texts.append(clean_text(text))
# print("Texts are complete.")


stories = list()
for i, text in enumerate(clean_summaries):
    stories.append({'story': text, 'highlights': clean_summaries[i]})# save to file
pickle.dump(stories, open('review_dataset.pkl', 'wb'))



