import base64
import io
import pandas as pd
import zipfile
from autogluon.tabular import TabularPredictor
from dash import html


# data loading function
def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return df
        elif "pkl" in filename:
            model = pd.read_pickle(io.BytesIO(decoded))
            if "<class 'flaml" in str(model.__class__).split("."):
                library = "Flaml"
            else:
                library = "AutoSklearn"
            return model, library
        elif "zip" in filename:
            with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
                zip_ref.extractall('./uploaded_model')

            model = TabularPredictor.load('./uploaded_model', require_py_version_match=False)
            library = "AutoGluon"
            return model, library

    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])
