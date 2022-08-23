# Text Segmentation on Datasets from Different (Dutch) Modalities
## MSc Reseasrch Project

### Setup

1. Create a new Python 3.8.* virtual environment and install the required packages.
  * Using Conda: `conda create -n ts_env python=3.8`
  * Using venv: `python3 -m venv ts_env`
2. Install the required packages: `pip install -r requirements.txt`
3. [Download GraphSeg](https://drive.google.com/file/d/14ikb3b3ZACsBGdY27hw8_tJ-fu8PqeCW/view?usp=sharing) and extract inside the Code folder.
4. Download the data and extract inside the Datasets folder.
  * [Dutch word2vec](https://github.com/clips/dutchembeddings), download the 160-dimensional combined vectors and extract the `word2vec-nl-combined-160.txt` file into the Datasets folder.
