# Recommending Songs and Playlists

Lucas De Oliveira, Chandrish Ambati, Anish Mukherjee

## Introduction and motivation

It all started with a dataset. In 2018, Spotify organized an Association for Computing Machinery (ACM) [RecSys Challenge](https://www.recsyschallenge.com/2018/) where they posted a [dataset of one million playlists](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge), challenging participants to recommend a list of 500 songs given a user-created playlist.

As both music lovers and data scientists, we were naturally drawn to this challenge. Right away, we agreed that combining song embeddings with some nearest-neighbors method for recommendation would likely produce very good results with not much effort. Importantly, we were curious about how a company like Spotify might do this recommendation task at scale – not with 1 million playlists but with the over [4 billion user-curated playlists](https://soundplate.com/how-many-playlists-are-there-on-spotify-and-other-spotify-stats/) on their platform. This realization raised serious questions about how to train a decent model since all that data would likely not fit in memory.

In this article we will discuss how we built a scalable ETL pipeline using Spark, MongoDB, Amazon S3, and Databricks to train a deep learning Word2Vec model to build song and playlist embeddings for recommendation. We’ll also see some visualizations we created on Tensorflow’s Embedding Projector.


## Workflow

### Collecting lyrics

The most tedious task of this project was collecting as many lyrics for the songs in the playlists as possible. We began by isolating the unique songs in the playlist files by their track URI; in total we had over 2 million unique songs. Then, we used the track name and artist name to look up the lyrics on the web. Initially, we used simple Python requests to pull in the lyrical information but this proved too slow for our purposes. We then used asyncio, which allowed us to make requests concurrently. This sped up the process significantly, reducing the downloading time of lyrics for 10k songs from 15 mins to under a minute. Ultimately, we were only able to collect lyrics for 138,000 songs.


### Preprocessing

The original dataset contains 1 million playlists spread across 1 thousand JSON files totaling about 33 GB of data. We used PySpark in Databricks to preprocess these separate JSON files into a single SparkSQL DataFrame and then joined this DataFrame with the lyrics we saved. From there, it was easy to read the files back from MongoDB into DataBricks to conduct our future analyses. 

Check out the [Preprocessing.ipynb](https://github.com/lbdeoliveira/song-playlist-recommendation/blob/main/notebooks/Preprocessing_Pipeline.ipynb) notebook to see how we preprocessed the data.


### Training song embeddings

For our analyses, we read our preprocessed SparkSQL DataFrame from MongoDB and grouped the records by playlist id, aggregating all of the songs in a playlist into a list under the column song_list. Below is a snapshot of the first five rows:

 <img width="629" alt="Screen Shot 2022-04-21 at 3 15 42 PM" src="https://user-images.githubusercontent.com/51177846/164567388-764b789a-2e1e-4c85-8ca8-9484e1557a08.png">

 
Using the Word2Vec model in Spark MLlib we trained song embeddings by feeding lists of track IDs from a playlist into to the model much like you would send a list of words from a sentence to train word embeddings. As shown below, we trained song embeddings in only 3 lines of PySpark code: 

 <img width="654" alt="Screen Shot 2022-04-21 at 3 16 03 PM" src="https://user-images.githubusercontent.com/51177846/164567415-d7886496-8665-404c-a22c-df37d75d1fe9.png">


We then saved the song embeddings down to MongoDB for later use. Below is a snapshot of the song embeddings DataFrame that we saved:

 <img width="704" alt="Screen Shot 2022-04-21 at 3 17 48 PM" src="https://user-images.githubusercontent.com/51177846/164567433-dd3870a9-c288-4441-8555-daa9e6cadfb7.png">

Check out the [Song_Embeddings.ipynb](https://github.com/lbdeoliveira/song-playlist-recommendation/blob/main/notebooks/Song_Embeddings.ipynb) notebook to see how we train song embeddings.


### Training playlist embeddings

Finally, we extended our recommendation task beyond simple song recommendation to recommending entire playlists. Given an input playlist, we would return the k closest or most similar playlists. We took a “continuous bag of songs” approach to this problem by calculating playlist embeddings as the average of all song embeddings in that playlist. 

This workflow started by reading back the song embeddings from MongoDB into a SparkSQL DataFrame. Then, we calculated a playlist embedding by taking the average of all song embeddings in that playlist and saved a playlist_id --> vector DataFrame in MongoDB. 

Check out the [Playlist_Embeddings.ipynb](https://github.com/lbdeoliveira/song-playlist-recommendation/blob/main/notebooks/Playlist_Embeddings.ipynb) notebook to see how we did this.


### Training lyrics embeddings

We trained lyrics embeddings by loading in a song's lyrics, separating the words into lists, and feeding those words to a Word2Vec model to produce 32-dimensional vectors for each word. We then took the average embedding across all words as that song's lyrical embedding. Ultimately, our analytical goal here was to determine whether users create playlists based on common lyrical themes by seeing if the pairwise song embedding distance and the pairwise lyrical embedding distance between two songs were correlated. Unsurprisingly, it appears they are not. 

Check out the [Lyrical_Embeddings.ipynb](https://github.com/lbdeoliveira/song-playlist-recommendation/blob/main/notebooks/Lyrical_Embeddings.ipynb) notebook to see our analysis.



## Notes on embedding training approach

You may be wondering why we used a language model (Word2Vec) to train these embeddings. Why not use a Pin2Vec or custom neural network model to predict implicit ratings? For practical reasons, we wanted to work exclusively in the Spark ecosystem and deal with the data in a distributed fashion. This was a constraint set on the project ahead of time and challenged us to think creatively. 

However, we found Word2Vec an attractive candidate model for theoretical reasons as well. The Word2Vec model uses a word’s context to train static embeddings by training the input word’s embeddings to predict its surrounding words. In essence, the embedding of any word is determined by how it co-occurs with other words. This had a clear mapping to our own problem: by using a Word2Vec model the distance between song embeddings would reflect the songs’ co-occurrence throughout 1M playlists, making it a useful measure for a distance-based recommendation (nearest neighbors). It would effectively model how people grouped songs together, using user behavior as the determinant factor in similarity. 

Additionally, the Word2Vec model accepts input in the form of a list of words. For each playlist we had a list of track IDs, which made working with the Word2Vec model not only conceptually but also practically appealing.


## Visualization and recommendation

After all of that, we were finally ready to visualize our results and make some interactive recommendations. We decided to represent our embedding results visually using Tensorflow’s Embedding Projector which maps the 32-dimensional song and playlist embeddings into an interactive visualization of a 3D embedding space. You have the choice of using PCA or tSNE for dimensionality reduction and cosine similarity or Euclidean distance for measuring distances between vectors.

Click [here](https://projector.tensorflow.org/?config=https://embedding-projector.s3.us-west-2.amazonaws.com/template_config_2M.json) for the song embeddings projector for the full 2 million songs, or [here](https://projector.tensorflow.org/?config=https://embedding-projector.s3.us-west-2.amazonaws.com/template_config.json) for a less crowded version with a random sample of 100k songs (shown below):


![songs_reduced](https://user-images.githubusercontent.com/51177846/164569070-88be5d94-a291-495d-b33d-3094634d9205.gif)


Click [here](https://projector.tensorflow.org/?config=https://embedding-projector.s3.us-west-2.amazonaws.com/template_config_playlist.json) for the playlist embeddings projector (shown below):


![playlists_reduced](https://user-images.githubusercontent.com/51177846/164569093-53affab8-1a62-408e-a5ef-b4ad3cd561cf.gif)


The neat thing about using Tensorflow’s projector is that it gives us a beautiful visualization tool and distance calculator all in one. Try searching on the right panel for a song and if the song is part of the original dataset, you will see the “most similar” songs appear under it.


## Conclusions

We were shocked by how this method of training embeddings actually worked. While the 2 million song embedding projector is crowded visually, we see that the recommendations it produces are actually quite good at grouping songs together.

Consider the embedding recommendation for The Beatles’ “Fool on the Hill”:

![Screen Shot 2022-04-15 at 5 46 55 PM](https://user-images.githubusercontent.com/51177846/164567966-5af2c500-031f-49f3-8f71-3b392b779b23.png)


Or the recommendation for Jay Z’s “Heart of the City (Ain’t No Love)”:

![Screen Shot 2022-04-15 at 5 52 22 PM](https://user-images.githubusercontent.com/51177846/164567916-28f6ebd6-0473-4d7b-b60c-55af628016f7.png)


Fan of Taylor Swift? Here are the recommendations for “New Romantics”:

![Screen Shot 2022-04-15 at 5 54 42 PM](https://user-images.githubusercontent.com/51177846/164567889-9ccf8326-7bc8-4c8e-9bc6-f123c0acba3f.png)



Secondly, we were delighted to find naturally occurring clusters in the playlist embeddings. Most notably, we see a cluster containing mostly Christian rock, one with Christmas music, one for reggaeton, and one large cluster where genres span its length rather continuously and intuitively.

Note also that when we select a playlist, we have many recommended playlists with the same names. This in essence validates our song embeddings. Recall that playlist embeddings were created by the taking the average embedding of all its songs; the name of the playlists did not factor in at all. That is, similar playlists are similar because they have similar, if not the same songs. The similar names only conceptually reinforce this fact.





