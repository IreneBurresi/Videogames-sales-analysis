<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://ireneburresi-videogames-sales-analysis-app-755m1w.streamlitapp.com">
    <img src="./assets/logo_joystick.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">VIDEOGAMES SALES ANALYSIS</h3>

  <p align="center">
   An in depth analysis of videogames sales all over the world plus a ML powered sales predictor
    <br />
    <a href="https://ireneburresi-videogames-sales-analysis-app-755m1w.streamlitapp.com"><strong> View demo »</strong></a>
    <br />
    <br />
    <a href="#about-this-project">About this project</a>
    ·
    <a href="#built-with">Built With</a>
    ·
    <a href="#getting-started">Getting started</a>
    ·
    <a href="#contact">Contact me</a>
  </p>
</div>

[screen]: ./assets/screen.png

<!-- ABOUT THE PROJECT -->
## About this project

[![Videogame sales][screen]](https://ireneburresi-videogames-sales-analysis-app-755m1w.streamlitapp.com)

This project analyses a dataset regarding the sales of most of the videogames published since '70s. 

In the Jupyter Notebook the dataset is:
1. loaded
2. cleaned and prepared for data visualisation
3. prepared for ML model training

Then 4 different ML models (one fro North America, one for Europe, one for Japan and one for all the other countries predictions) are trained with  an ensemble method.
A Grid Search allows to choose te best models, that are then used to predct values on the web app.
The web app is deployed using Streamlit (at https://ireneburresi-videogames-sales-analysis-app-755m1w.streamlitapp.com).

### About Dataset
The dataset contains a list of video games with sales greater than 100,000 copies. 

It was generated by a scrape of vgchartz.

Fields include
* Rank - Ranking of overall sales
* Name - The games name
* Platform - Platform of the games release (i.e. PC,PS4, etc.)
* Year - Year of the game's release
* Genre - Genre of the game
* Publisher - Publisher of the game
* NA_Sales - Sales in North America (in millions)
* EU_Sales - Sales in Europe (in millions)
* JP_Sales - Sales in Japan (in millions)
* Other_Sales - Sales in the rest of the world (in millions)
* Global_Sales - Total worldwide sales.


There are 16,598 records.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This project is built in:
<div align="center">

[![Python][Python-shield]][Python-url]
</div>

Using the following libraries:

<div align="center">

[![NumPy][NumPy-shield]][NumPy-url]      [![Pandas][Pandas-shield]][Pandas-url]    
</div>

The analysis and model training is done in:

<div align="center">

[![Jupyter][Jupyter-shield]][Jupyter-url]

</div>

Powered by:
<div align="center">

[![Colab][Colab-shield]][Colab-url]

</div>

Deployed with:
<div align="center">
  
[![Streamlit][Streamlit-shield]][Streamlit-url]
  
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Create a conda environment
```
 conda create -p venv python==3.7 -y
```
Activate conda environement
```
conda activate venv
```
Install requirements.txt file using below command
```
pip install -r requirements.txt
```


### Download

After the download, simply unzip the directory and then move to the project directory. To open the app in streamlit:
```
streamlit run app.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact
You can find me at:
burresi.irene@icloud.com - Irene Burresi

Have a look at my personal resume site here

<p align="right">(<a href="#readme-top">back to top</a>)</p>



[Python-shield]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue
[Python-url]: https://www.python.org
[NumPy-shield]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org
[Pandas-shield]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org
[Plotly-shield]: https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://pandas.pydata.org
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org
[SciPy-shield]: https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white
[SciPy-url]: https://scipy.org
[Streamlit-shield]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io
[Tensorflow-shield]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org
[MacOS-shield]: https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white
[Linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[Linkedin-url]: https://www.linkedin.com/in/ireneburresi/
[Kaggle-shield]: https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white
[Jupyter-shield]:	https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org
[PowerBI-shield]: https://img.shields.io/badge/PowerBI-F2C811?style=for-the-badge&logo=Power%20BI&logoColor=white
[Colab-shield]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252
[Colab-url]: https://colab.research.google.com
[PyCharm-shield]: https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white
[Tableau-shield]: https://img.shields.io/static/v1?style=for-the-badge&message=Tableau&color=E97627&logo=Tableau&logoColor=FFFFFF&label=
[Tableau-url]: https://www.tableau.com
