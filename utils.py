import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from wordcloud import WordCloud


def preprocess_sentence():
	""" Load spacy and return preprocessing function.
	"""
	nlp = spacy.load('en_core_web_md')

	def preprocess_sentence_(sentence):
		""" Preprocessing function: remove stop words and punctuation.
		"""
		return ' '.join([token.lemma_ for token in nlp(sentence) \
				if not token.is_stop and not token.is_punct])

	return preprocess_sentence_


def process_data(df, key, path=None):
	""" Clean and dump motivations from a dataframe.
	"""
	# Extract columns.
	data = df[key]

	# Preprocess each sentence.
	processed_data = data.apply(preprocess_sentence())

	# Drop potential empty strings due to preprocessing.
	data           = data[processed_data != '']
	processed_data = processed_data[processed_data != '']

	# Create new DataFrame.
	df = pd.DataFrame()
	df[key] = data
	df[f'processed_{key}'] = processed_data

	if path:
		df.to_csv(path, index=False)

	return df


def print_wordcloud(top_words, scores, topic_index, topic_name, num_topics):
	""" Plot wordcloud with top scoring words.
	"""
	freq = {}
	for word, score in zip(top_words, scores):
		freq[word] = int(score)

	# Create and generate a word cloud image.
	wordcloud = WordCloud(background_color='white').generate_from_frequencies(freq)

	# Display the generated image.
	plt.subplot(np.ceil(num_topics / 2.).astype(np.int), 2, topic_index + 1)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.title(topic_name)
	plt.axis("off")

