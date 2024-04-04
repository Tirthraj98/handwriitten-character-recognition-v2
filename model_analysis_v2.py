import os
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image #,ImageEnhance
# from tensorflow.keras.preprocessing import image
import joblib
import cv2
import time

df = pd.read_csv("barakshari.csv",index_col=0)

gujarati_consonants_dict = {
    'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ', 'ng': 'ઙ',
    'ch': 'ચ', 'chh': 'છ', 'j': 'જ', 'z': 'ઝ',
    'at': 'ટ', 'ath': 'ઠ', 'ad': 'ડ', 'adh': 'ઢ', 'an': 'ણ',
    't': 'ત', 'th': 'થ', 'd': 'દ', 'dh': 'ધ', 'n': 'ન',
    'p': 'પ', 'f': 'ફ', 'b': 'બ', 'bh': 'ભ', 'm': 'મ',
    'y': 'ય', 'r': 'ર', 'l': 'લ', 'v': 'વ', 'sh': 'શ',
    'shh': 'ષ', 's': 'સ', 'h': 'હ', 'al': 'ળ', 'ks': 'ક્ષ',
    'gn': 'જ્ઞ'
}

gujarati_vowels_dict = {'a': 'આ', 'i': 'ઇ', 'ee': 'ઈ', 'u': 'ઉ',
    'oo': 'ઊ', 'ri': 'ઋ', 'rii': 'ૠ', 'e': 'એ', 'ai': 'ઐ',
    'o': 'ઓ', 'au': 'ઔ', 'amn': 'અં', 'ah': 'અઃ',"ru" : "અૃ","ra" : "અ્ર",
    'ar' : "્રઅ"
}



def get_gujarati_label(class_label, gujarati_dict):
    guj_class_label = ""
    if class_label.lower() in gujarati_dict.keys():
        guj_class_label = gujarati_dict[class_label.lower()]

    return guj_class_label

# Function to load and preprocess images
def load_image(img_path, target_size=(50, 50)):
    image = Image.open(img_path)
    image = image.resize((50, 50))
    image = image.convert('L')
    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # image = Image.fromarray(image)
    # image = adjust_contrast_brightness(image)
    # image = np.array(image)
    # image = cv2.fastNlMeansDenoisingColored(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # image = remove_extra_spaces(image,padding=2)
    image_array = image / 255.0
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    image_array = np.expand_dims(image_array, axis=0)
    # image_array = np.expand_dims(image, axis=-1)
    # print(image_array.shape)
    return image_array

# Function to predict class and confidence
def predict_class(model, encoder, img):
    st = time.time()
    pred = model.predict(img,verbose=0)
    et = time.time()
    inf_time = et-st
    class_index = np.argmax(pred)
    class_label = encoder.inverse_transform([class_index])[0]
    confidence = pred[0][class_index]
    return class_label, confidence, inf_time

def is_image_file(file_path):

    # Get the file extension
    _, file_extension = os.path.splitext(file_path)

    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # Check if the file extension is in the list of image extensions
    if file_extension.lower() in image_extensions:
        return True
    else:
        return False
    
# Define the function analyze_character
def analyze_character(character):
    # Define sets for consonants and vowels
    consonants = ['K', 'KH', 'G', 'GH', 'CH', 'CHH', 'J', 'Z', 'AT', 'ATH', 'AD', 'ADH', 'AN', 'T', 'TH', 'D', 'DH', 'N', 'P', 'F', 'B', 'BH', 'M', 'Y', 'R', 'L', 'V', 'SH', 'SHH', 'S', 'H', 'AL', 'KS', 'GN']
    vowels = ["A", "I", "EE", "U", "OO", "E", "O", "AI", "AU", "AMN", "AH", "RA", "AR", "RU"]

    # Initialize variables to store consonant and vowel parts
    consonant_part = ""
    vowel_part = ""

    # Check each possible length of the consonant part
    for i in range(1, len(character)):
        if not character[:2] == "AR":
            if character[:i] in consonants and character[i:] in vowels:
                consonant_part = character[:i]
                vowel_part = character[i:]
                break
        else:
            vowel_part = character[:2]
            consonant_part = character[2:-1]


    return consonant_part, vowel_part

# Load models
character_model = load_model("/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/models/colab_models/v2_models/Character_model_gray_v2.h5")
consonant_model = load_model("/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/models/colab_models/v2_models/Consonant_model_gray_v2.h5")
vowel_model = load_model("/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/models/colab_models/v2_models/Vowel_model_gray_v2.h5")

character_encoder = joblib.load("/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/models/colab_models/v2_models/Character_label_encoder_gray_v2.joblib")
consonant_encoder = joblib.load("/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/models/colab_models/v2_models/Consonant_label_encoder_gray_v2.joblib")
vowel_encoder = joblib.load("/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/models/colab_models/v2_models/Vowel_label_encoder_gray_v2.joblib")

# Directory containing images
base_dir = "/home/salogosm/code/machine learning with python/OpenCV projects/Mam's ML project/final_project/Original_dataset/final_dataset_RGB_50x50_115545"

# Get folder names
folders = os.listdir(base_dir)
folders = sorted(folders)

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['True_label_en',
                                    'Predicted_character_label_en','character_conf','character_inf_time',
                                    'Predicted_vowel_label_en','vowel_conf','vowel_inf_time',
                                    'Predicted_consonant_label_en','consonant_conf','consonant_inf_time',
                                    'combined_character_guj_label',
                                    'True_predicted_characters','True_predicted_vowels','True_predicted_consonants','Total_samples'])

# Iterate through folders
for index,folder in enumerate(folders):
    folder_path = os.path.join(base_dir, folder)
    if index<400:
        continue
    # elif index>400:
    #     break
    if os.path.isdir(folder_path):
        image_files = os.listdir(folder_path)
        # for img_file in image_files:
        if image_files:
            avg_vowel_conf = 0
            avg_consonant_conf = 0
            avg_character_conf = 0
            avg_vowel_inf_time = 0
            avg_consonant_inf_time = 0
            avg_character_inf_time = 0
            True_characters = 0
            True_vowels = 0
            True_consonants = 0
            pred_chars = []
            pred_cons = []
            pred_vows = []
            for img in range(len(image_files)):
                img_path = os.path.join(folder_path, image_files[img])
                if is_image_file(file_path=img_path):
                    img = load_image(img_path)
                    # Predictions
                    character_pred, character_conf, character_inf_time = predict_class(character_model,character_encoder, img)
                    consonant_pred, consonant_conf,consonant_inf_time = predict_class(consonant_model,consonant_encoder, img)
                    vowel_pred, vowel_conf,vowel_inf_time = predict_class(vowel_model,vowel_encoder, img)
                    consonant_guj_label = get_gujarati_label(consonant_pred, gujarati_consonants_dict)
                    vowel_guj_label = get_gujarati_label(vowel_pred, gujarati_vowels_dict)
                    # print("Consonant label:", consonant_pred)
                    # print("Vowel label:", vowel_pred)
                    # print("Consonant label:", consonant_guj_label)
                    # print("Vowel label:", vowel_guj_label)
                    combined_character_guj_label = df.loc[consonant_guj_label.strip(), vowel_guj_label.strip()]
                    # Predicted_label_guj = df.loc[get_gujarati_label(analyze_character(character_pred)[0],gujarati_consonants_dict).strip(), get_gujarati_label(analyze_character(character_pred)[1],gujarati_vowels_dict).strip()]

                    # appending each predicted for calculating maximun correct
                    pred_chars.append(character_pred)
                    pred_vows.append(vowel_pred)
                    pred_cons.append(consonant_pred)

                    # True labels
                    True_consonant_en, True_vowel_en = analyze_character(folder)
                    # True_consonant_guj = get_gujarati_label(True_consonant_en, gujarati_consonants_dict)
                    # True_vowel_guj = get_gujarati_label(True_vowel_en, gujarati_vowels_dict)
                    # True_character_label_guj = df.loc[True_consonant_guj.strip(), True_vowel_guj.strip()]

                    # calculate how many correct
                    if folder==character_pred:
                        True_characters += 1
                    if True_consonant_en==consonant_pred:
                        True_consonants += 1
                    if True_vowel_en==vowel_pred:
                        True_vowels += 1

                    # sum of confidence of all images
                    avg_vowel_conf += vowel_conf
                    avg_consonant_conf += consonant_conf
                    avg_character_conf += character_conf

                    # sum of inference time of all images
                    avg_vowel_inf_time += vowel_inf_time
                    avg_consonant_inf_time += consonant_inf_time
                    avg_character_inf_time += character_inf_time
                else:
                    print(f"{img_path} is not image path.")
                
            max_char = max(pred_chars, key=lambda x: pred_chars.count(x))
            max_vow = max(pred_vows, key=lambda x: pred_vows.count(x))
            max_con = max(pred_cons, key=lambda x: pred_cons.count(x))
            # max_con_guj = get_gujarati_label(max_con, gujarati_consonants_dict)
            # max_vow_guj = get_gujarati_label(max_vow, gujarati_vowels_dict)
            # max_char_guj = df.loc[get_gujarati_label(analyze_character(max_char)[0],gujarati_consonants_dict).strip(), get_gujarati_label(analyze_character(max_char)[1],gujarati_vowels_dict).strip()]

            results_df = pd.concat([results_df, pd.DataFrame({'True_label_en': folder,
                                        'Predicted_character_label_en':max_char,'character_conf':avg_character_conf/len(image_files),'character_inf_time':avg_character_inf_time/len(image_files),
                                        'Predicted_vowel_label_en':max_vow, 'vowel_conf':avg_vowel_conf/len(image_files),'vowel_inf_time':avg_vowel_inf_time/len(image_files),
                                        'Predicted_consonant_label_en':max_con, 'consonant_conf':avg_consonant_conf/len(image_files),'consonant_inf_time':avg_consonant_inf_time/len(image_files),
                                        'combined_character_guj_label':combined_character_guj_label,
                                        'True_predicted_characters':True_characters,'True_predicted_vowels':True_vowels,'True_predicted_consonants':True_consonants,
                                        'Total_samples':len(image_files)
                                        },
                                        index=[0])], ignore_index=True)
        else:
            print(f"\n\nthis {folder} gives error!\n\n")
            # print(folder)
            # print(folder)
            # print(folder)
            # print(folder)
    else:
        print(f"\n\nthis folder named {folder} is not availbale!!!\n\n")
print(index)

# os.chdir("/content/drive/MyDrive/R_Phd_project/HP_tuning_results")

# Save results to CSV
results_df.to_csv('model_evaluation_results_S_iter_gray_last_400to_.csv', index=False)
# results_df.to_csv('Trial_10.csv', index=False)

print('File saved successfully!!')