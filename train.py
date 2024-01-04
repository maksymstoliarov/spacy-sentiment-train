import spacy
import random
# import re
import pandas as pd
from spacy.pipeline.textcat import Config, single_label_cnn_config, single_label_bow_config, single_label_default_config
from spacy.training.example import Example
from spacy.util import minibatch
from sklearn.model_selection import train_test_split

# spacy.prefer_gpu()
# spacy.require_gpu()


# def clean_text(text):
#     text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
#
#     return text

# read xlsx file
df = pd.read_excel('datasets/dataset2.xlsx')
df.columns=['text', 'sentiment']
df['sentiment'].value_counts()

df, test = train_test_split(df, train_size=0.9, shuffle=True, stratify=df['sentiment'])

train_texts = df['text'].values
train_labels = [{'cats': {'positive': label == 'Позитивний',
                          'negative': label == 'Негативний'}}
                for label in df['sentiment']]

train_data = list(zip(train_texts, train_labels))
print(len(train_data))

# print(train_data)
# exit()

# Create an empty model
nlp = spacy.blank("uk")
config = Config().from_str(single_label_bow_config)
text_cat = nlp.add_pipe('textcat', config=config, last=True)
text_cat.add_label("positive")
text_cat.add_label("negative")

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(25):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        texts, annotations = zip(*batch)

        example = []
        # Update the model with iterating each text
        for i in range(len(texts)):
            doc = nlp.make_doc(texts[i])
            example.append(Example.from_dict(doc, annotations[i]))

            # Update the model
        nlp.update(example, drop=0.5, losses=losses)
    print(losses)

nlp.to_disk('models/sentiment_model')

print("Model trained and saved")
exit()

text_nlp = spacy.load("models/sentiment_model")

test_texts = test['text']
test_te = list(test_texts)

print("Enter:")
input_text = input()
doc = text_nlp(input_text)

print(doc.cats)







