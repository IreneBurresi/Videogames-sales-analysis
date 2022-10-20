# ===========================================================================================================
#                                               DIGITAL RESUME
# ===========================================================================================================

# ================================================= IMPORTS =================================================
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import joblib


# ============================================== PATH SETTINGS ==============================================
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

# ============================================ GENERAL SETTINGS =============================================

# The title and icon showed on the browser tab
PAGE_TITLE = "Videogames Sales"
PAGE_ICON = ":wave:"

# Social media
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/ireneburresi",
    "GitHub": "https://github.com/IreneBurresi",
}

github_link = 'https://github.com/IreneBurresi'
github_image = 'https://img.icons8.com/ios-glyphs/30/000000/github.png'

linkedin_link = "https://www.linkedin.com/in/ireneburresi"
linkedin_image = "https://img.icons8.com/ios/30/000000/linkedin-2--v1.png"


# Method to show social media links as clickable images
def add_social_media_links():
    st.write('[![Github](' + github_image + ')](' + github_link + ')'
             + ' [![Linkedin](' + linkedin_image + ')](' + linkedin_link + ')')


# ================================================ WEBSITE =================================================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# Loading css
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# ------------------------------------------------- HEADER -------------------------------------------------
st.title("Videogames Sales")
st.write("By Irene Burresi")


# -------------------------------------------- NAVIGATION MENU ---------------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Sales predictor", "Data Visualisation"],
    #icons=["emoji-smile", "mortarboard", "code-slash", "paperclip"],
    orientation="horizontal",
)

# --------------------------------------------- SALES PREDICTOR ---------------------------------------------
genre = "Action"
console_company = "PlayStation"
publisher = "Electronic Arts"
year = 2021

global_dataset = pd.read_csv("./data/videogame_sales_cleaned.csv")
genre_list = global_dataset["Genre"].value_counts().index
publisher_list = global_dataset["Publisher"].value_counts().index
console_company_list = global_dataset["Console_Company"].value_counts().index
clf_EU = joblib.load("./models/model_EU.pkl")
clf_NA = joblib.load("./models/model_NA.pkl")
clf_JP = joblib.load("./models/model_JP.pkl")
clf_other = joblib.load("./models/model_other.pkl")
HtmlFile = open("./pandas_profiling-2.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()

def get_data() -> pd.DataFrame:
    return pd.read_csv("./data/videogame_sales_cleaned.csv")


if selected == "Sales predictor":
    st.subheader("ML Powered Sales predictor")
    st.write("Select videogame informations:")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        genre = st.selectbox("Genre", genre_list)
        console_company = st.selectbox("Console", console_company_list)
    with col2:
        publisher = st.selectbox("Publisher", publisher_list)
        year = st.slider("Year", min_value=1970, max_value=2022, value=2020, step=1)
    st.write(
        "Predict the sales of a " + genre + " videogame, published by " + publisher + " for the " + console_company + " console in " + str(
            year) + "?")
    st.write("Where?")
    col3, col4 = st.columns(2, gap="small")
    with col3:
        EU_sales = st.checkbox("EU sales")
        NA_sales = st.checkbox("North America sales")
    with col4:
        JP_sales = st.checkbox("Japan sales")
        other_sales = st.checkbox("Other countries sales")

    global_sales = st.checkbox("Global sales")

    submit = st.button("Predict")
    if submit:
        data = {'Genre': genre,
                'Console_Company': console_company,
                'Publisher': publisher,
                'Year': year}
        features = pd.DataFrame(data, index=[0])
        if(EU_sales==True or global_sales==True):
            prediction_EU = round(number=(clf_EU.predict(features)[0] *1000000))
            st.write("The videogame is estimed to sell " + str(prediction_EU) + " copies in Europe")
        if (NA_sales == True or global_sales == True):
            prediction_NA = round(number=(clf_NA.predict(features)[0] * 1000000))
            st.write("The videogame is estimed to sell " + str(prediction_NA) + " copies in North America")
        if (JP_sales == True or (global_sales == True)):
            prediction_JP = round(number=(clf_JP.predict(features)[0] * 1000000))
            st.write("The videogame is estimed to sell " + str(prediction_JP) + " copies in Japan")
        if (other_sales == True or global_sales == True):
            prediction_other = round(number=(clf_other.predict(features)[0] * 1000000))
            st.write("The videogame is estimed to sell " + str(prediction_other) + " copies in other countries")
elif selected == "Data Visualisation":
    st.write("### Pandas Profile")
    st.write("---")
    st.markdown(source_code, unsafe_allow_html=True)
    #components.html(source_code, height = 600)



