import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r"MODEL_PATH")

# Class labels
class_labels = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy'
]
# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make a prediction
def predict_image(file_path):
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    fungicide_links = {
    'Apple__Apple_scab': 'https://www.prestonhardware.com/product/king-32723-fungicide-crystal-rotten-egg-yellow-500-g/',
    'Apple_Black_rot': 'https://www.boutiquesignegarneau.com/bioprotec-fongicide-arbre-fruitiers-conc-500ml.html',
    'Apple_Cedar_apple_rust': 'https://houseofplants.ca/products/safers-sulphur-dust-fungicide-and-miticide-300g?variant=44144088809724&country=CA&currency=CAD',
    'Apple__healthy': None,  
    'Blueberry__healthy': None,
    'Cherry(including_sour)_healthy': None,
    'Raspberry_healthy': None,
    'Soybean__healthy': None,
    'Cherry(including_sour)Powdery_mildew': 'https://www.cropscience.bayer.ca/en/products/fungicides/luna-products/luna-sensation',
    'Corn(maize)Common_rust': 'https://www.amazon.com/Southern-Ag-Liquid-Copper-Fungicide/dp/B0BZR8M8M4/ref=sr_1_1?dib=eyJ2IjoiMSJ9.twYEOljRWgtsTS6T-VOPkSSRn98jgC4ZWOFkPTFuQ9_GjHj071QN20LucGBJIEps.dWT07Lmh-Y8_cKaKdZQ_6glUdFtWZXmTmsuibWV8ohI&dib_tag=se&keywords=Southern-AG-Liquid-Copper-Fungicide&qid=1733116014&sr=8-1&th=1', #--
    'Corn_(maize)Northern_Leaf_Blight': 'https://www.amazon.com/Daconil%C2%AE-Fungicide-Concentrate-Insects-oz/dp/B0DC7GRJWD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.WWs1uqZ_qJOT94IRmPqcr5aOOOCdGiSoYdZQvZQ5D1BUrB4-mFSsDQLFxSQFoloQcFjqaHBYSBZJBtgO4PAYygXMBb7uBB9RMAHJZe10j0Az-iLaHR62gbh4sGSg_ZZW4ZCjYdAY_9eeAAa9xsrAI5hH3EdjwslFZUIVBgpFSbs.HFOo9vJGQ-nvBVYGrGFhpM_dvWegULdBtpVXsoPl8oY&dib_tag=se&keywords=daconil%2Bfungicide%2Bconcentrate&qid=1733116056&sr=8-1&th=1', #--
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot': 'https://www.dkhardware.com/quali-pro-83013365-quali-pro-qt-ppz-143-fungicide-product-4102236.html',
    'Corn(maize)healthy': None,
    'Grape__Black_rot': 'https://www.amazon.com/Bonide-811-Copper-Fungicide-32/dp/B000BWY3OQ/',
    'Grape__Esca(Black_Measles)': 'https://www.dkhardware.com/southern-ag-retail-01600-captan-fungicide-8-oz-product-7871947.html?utm_source=google&utm_medium=shopping&utm_campaign=free_listings', #--
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 'https://www.amazon.com/Bonide-811-Copper-Fungicide-32/dp/B000BWY3OQ/',
    'Grape___healthy': None,
    'Orange__Haunglongbing(Citrus_greening)': None,   
    'Peach__Bacterial_spot': 'https://www.amazon.com/Southern-Ag-Liquid-Copper-Fungicide/dp/B0BZR8M8M4/ref=sr_1_1?dib=eyJ2IjoiMSJ9.twYEOljRWgtsTS6T-VOPkSSRn98jgC4ZWOFkPTFuQ9_GjHj071QN20LucGBJIEps.dWT07Lmh-Y8_cKaKdZQ_6glUdFtWZXmTmsuibWV8ohI&dib_tag=se&keywords=Southern-AG-Liquid-Copper-Fungicide&qid=1733116212&sr=8-1&th=1', #--
    'Peach__healthy': None,
    'Pepper,bell_Bacterial_spot': 'https://www.amazon.com/Bonide-811-Copper-Fungicide-32/dp/B000BWY3OQ/', #--
    'Pepper,_bell_healthy': None,
    'Potato__Early_blight': 'https://www.amazon.com/Bonide-811-Copper-Fungicide-32/dp/B000BWY3OQ/',
    'Potato__Late_blight': 'https://www.amazon.com/Daconil%C2%AE-Fungicide-Concentrate-Insects-oz/dp/B0DC7GRJWD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.WWs1uqZ_qJOT94IRmPqcr5aOOOCdGiSoYdZQvZQ5D1BUrB4-mFSsDQLFxSQFoloQcFjqaHBYSBZJBtgO4PAYygXMBb7uBB9RMAHJZe10j0Az-iLaHR62gbh4sGSg_ZZW4ZCjYdAY_9eeAAa9xsrAI5hH3EdjwslFZUIVBgpFSbs.HFOo9vJGQ-nvBVYGrGFhpM_dvWegULdBtpVXsoPl8oY&dib_tag=se&keywords=daconil%2Bfungicide%2Bconcentrate&qid=1733116056&sr=8-1&th=1', #--
    'Potato_healthy': None,
    'Raspberry_healthy': None,
    'Soybean__healthy': None,
    'Squash__Powdery_mildew': 'https://www.dkhardware.com/spectracide-hg-96203-acre-plus-triazicide-insect-killer-for-lawns-landscapes-1-gallon-concentrate-product-discontinued-5718383.html?utm_source=google&utm_medium=shopping&utm_campaign=free_listings&nbt=nb%3Aadwords%3Ax%3A18286622386%3A%3A&nb_adtype=pla&nb_kwd=&nb_ti=&nb_mi=2663876&nb_pc=online&nb_pi=5718383&nb_ppi=&nb_placement=&nb_li_ms=&nb_lp_ms=&nb_fii=&nb_ap=&nb_mt=&gad_source=1', #--
    'Strawberry_Leaf_scorch': 'https://www.amazon.com/Bonide-811-Copper-Fungicide-32/dp/B000BWY3OQ/',
    'Strawberry__healthy': None,
    'Tomato__Bacterial_spot': 'https://www.amazon.com/Southern-Ag-Liquid-Copper-Fungicide/dp/B0BZR8M8M4/ref=sr_1_1?dib=eyJ2IjoiMSJ9.twYEOljRWgtsTS6T-VOPkSSRn98jgC4ZWOFkPTFuQ9_GjHj071QN20LucGBJIEps.dWT07Lmh-Y8_cKaKdZQ_6glUdFtWZXmTmsuibWV8ohI&dib_tag=se&keywords=Southern-AG-Liquid-Copper-Fungicide&qid=1733116212&sr=8-1&th=1', #--
    'Tomato_Early_blight': 'https://www.amazon.com/Daconil%C2%AE-Fungicide-Concentrate-Insects-oz/dp/B0DC7GRJWD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.WWs1uqZ_qJOT94IRmPqcr5aOOOCdGiSoYdZQvZQ5D1BUrB4-mFSsDQLFxSQFoloQcFjqaHBYSBZJBtgO4PAYygXMBb7uBB9RMAHJZe10j0Az-iLaHR62gbh4sGSg_ZZW4ZCjYdAY_9eeAAa9xsrAI5hH3EdjwslFZUIVBgpFSbs.HFOo9vJGQ-nvBVYGrGFhpM_dvWegULdBtpVXsoPl8oY&dib_tag=se&keywords=daconil%2Bfungicide%2Bconcentrate&qid=1733116056&sr=8-1&th=1', #--
    'Tomato_Late_blight': 'https://www.amazon.com/Daconil%C2%AE-Fungicide-Concentrate-Insects-oz/dp/B0DC7GRJWD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.WWs1uqZ_qJOT94IRmPqcr5aOOOCdGiSoYdZQvZQ5D1BUrB4-mFSsDQLFxSQFoloQcFjqaHBYSBZJBtgO4PAYygXMBb7uBB9RMAHJZe10j0Az-iLaHR62gbh4sGSg_ZZW4ZCjYdAY_9eeAAa9xsrAI5hH3EdjwslFZUIVBgpFSbs.HFOo9vJGQ-nvBVYGrGFhpM_dvWegULdBtpVXsoPl8oY&dib_tag=se&keywords=daconil%2Bfungicide%2Bconcentrate&qid=1733116056&sr=8-1&th=1', #--
    'Tomato__Leaf_Mold': 'https://www.dkhardware.com/spectracide-hg-96203-acre-plus-triazicide-insect-killer-for-lawns-landscapes-1-gallon-concentrate-product-discontinued-5718383.html?utm_source=google&utm_medium=shopping&utm_campaign=free_listings&nbt=nb%3Aadwords%3Ax%3A18286622386%3A%3A&nb_adtype=pla&nb_kwd=&nb_ti=&nb_mi=2663876&nb_pc=online&nb_pi=5718383&nb_ppi=&nb_placement=&nb_li_ms=&nb_lp_ms=&nb_fii=&nb_ap=&nb_mt=&gad_source=1',#--
    'Tomato__Septoria_leaf_spot': 'https://www.amazon.com/Bonide-811-Copper-Fungicide-32/dp/B000BWY3OQ/',
    'Tomato_Spider_mites Two-spotted_spider_mite': 'https://www.amazon.com/Safer-Brand-End-All-Insect-Killer/dp/B004PBCFG2/',
    'Tomato__Target_Spot': 'https://www.amazon.com/Daconil%C2%AE-Fungicide-Concentrate-Insects-oz/dp/B0DC7GRJWD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.WWs1uqZ_qJOT94IRmPqcr5aOOOCdGiSoYdZQvZQ5D1BUrB4-mFSsDQLFxSQFoloQcFjqaHBYSBZJBtgO4PAYygXMBb7uBB9RMAHJZe10j0Az-iLaHR62gbh4sGSg_ZZW4ZCjYdAY_9eeAAa9xsrAI5hH3EdjwslFZUIVBgpFSbs.HFOo9vJGQ-nvBVYGrGFhpM_dvWegULdBtpVXsoPl8oY&dib_tag=se&keywords=daconil%2Bfungicide%2Bconcentrate&qid=1733116056&sr=8-1&th=1', #---
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': None, 
    'Tomato_Tomato_mosaic_virus': None,  
    'Tomato__healthy': None
    }

    fungicide_link = fungicide_links.get(predicted_class, '#')
    return predicted_class, confidence, fungicide_link

pred_class, conf, link = predict_image(r"IMAGE_PATH")
print("Predicted Class:" + pred_class)
print("Confidence:" + conf)
print("Treatment:" + link)