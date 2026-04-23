# Import required libraries
import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle

# Molecular descriptor calculator
def desc_calc():
    cmd = [
        "java", "-Xms2G", "-Xmx2G", "-Djava.awt.headless=true",
        "-jar", "./PaDEL-Descriptor/PaDEL-Descriptor.jar",
        "-removesalt", "-standardizenitro", "-fingerprints",
        "-descriptortypes", "./PaDEL-Descriptor/PubchemFingerprinter.xml",
        "-dir", "./", "-file", "descriptors_output.csv"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')
    if process.returncode != 0 or not os.path.exists('descriptors_output.csv'):
        raise RuntimeError(
            "PaDEL-Descriptor failed. Ensure:\n"
            "1. Java is installed and on your PATH\n"
            "2. PaDEL-Descriptor/ folder is extracted inside the App/ directory\n\n"
            f"stderr: {error.decode(errors='replace')}"
        )

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # string to byte conversion
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data, molecule_name):
    model = pickle.load(open('aromatase_model.pkl', 'rb'))
    prediction = model.predict(input_data)
    st.header('**Prediction Output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(molecule_name, name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Page title
st.markdown("""
    # Bioactivity Prediction App (Aromatase)

    This app allows you to predict the bioactivity towards inhibiting the `Aromatase` enzyme, which is a drug target for breast cancer.
""")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("Use `example_aromatase.txt` found in the App folder as a test input.")

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

    st.header('**Original Input Data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        try:
            desc_calc()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset, load_data[1])
else:
    st.info('Upload input data in the sidebar to start!')