# Text Segmentation on Datasets from Different (Dutch) Modalities
## MSc Research Project

### Setup

1. Create a new Python 3.8.* virtual environment and install the required packages.
   * Using Conda: `conda create -n ts_env python=3.8`
   * Using venv: `python3 -m venv ts_env`
2. Install the required packages: `pip install -r requirements.txt`
3. [Download GraphSeg](https://drive.google.com/file/d/14ikb3b3ZACsBGdY27hw8_tJ-fu8PqeCW/view?usp=sharing) and extract inside the Code folder.
   * Can also be cloned from the [original repository](https://bitbucket.org/gg42554/graphseg/src/master/), though this only contains a .jar file for English.
4. Download the data and extract it directly into the Datasets folder.
   * [Dutch word2vec](https://github.com/clips/dutchembeddings), download the 160-dimensional combined embeddings and extract the `word2vec-nl-combined-160.txt` file into the Datasets folder.
   * [ENWiki](https://drive.google.com/file/d/13UBi6n5PabD9HaZHLn51QBnhV_MF_0da/view?usp=sharing)
   * [NLWiki](https://drive.google.com/file/d/15YD3gmZVSe2-sdxPFSdDJMEjnjzoxdqM/view?usp=sharing)
   * [NLNews](https://drive.google.com/file/d/1X0ZhqI1ojH__BdRKpCH4FDviJcC6rQ6t/view?usp=sharing)
5. **OR** create the datasets yourself by following the Python Notebooks in the Code/data_extractors folder.
