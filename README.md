# A Linguistic Map of Nationally Determined Contributions
Final project for Jean-Philippe COINTET's Social Media Analysis course by Dominy GALLO, Chiara HAMPTON, Maëlle LEFEUVRE, Camille LEFEVRE, and Chloe PRYCE.

Date: May 8, 2025

## Introduction

GRAPH STYLING: 

Title: Average Normalized Count of "[Word]” by Income Level/Sub-region etc
x-label: Average "[Word]" Count

MAP STYLING

Normalized Count of '[Word]' by Country


## Data Processing

For the purposes of this project, we have used the Nationally Determined Contributions (NDCs) provided in the UNFCCC Secretariat's database as a point of analysis. From this, we have approximately 200 datapoints, ranging from various countries and regions from around the world.

Website: [https://unfccc.int/NDCRE] 

There were however some challenges in obtaining these files in a way where analysis could be made, through either word frequency, topic modelling, counting words or experimenting with regressions. Thus, to allow our readers to potentially replicate this, we have included here our code to access these resources:

### Code:

    #Download the following
    import os
    print(os.listdir())

    #Download each file pdf individually, then upload them to google drive in a file that is accessible. Then mount your drive.
    from google.colab import drive
    drive.mount('/content/drive')
    
    #Next step, is converting the files to txt, through the following code:
    !pip install pymupdf
    import fitz  # PyMuPDF
    
    os.makedirs("ndc_txts", exist_ok=True)
    
    for pdf_file in os.listdir("ndc_pdfs"):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join("ndc_pdfs", pdf_file)
            txt_path = os.path.join("ndc_txts", pdf_file.replace(".pdf", ".txt"))
    
            try:
                doc = fitz.open(pdf_path)
                text = "".join([page.get_text() for page in doc])
                doc.close()
    
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
    
                print(f"✅ Converted: {pdf_file}")
            except Exception as e:
                print(f"❌ Error converting {pdf_file}: {e}")

    #Then, create a source path to input all of these txts into an accessible folder, so you may refer back to it later through the process.

    import shutil
    import os
    
    # Source path (your Colab folder)
    source_folder = "ndc_txts"
    
    # Destination path (on your Google Drive)
    destination_folder = "/content/drive/MyDrive/Name_of_Folder"  # Change 'MyDrive/ndc_texts' to a different path each time!
    
    # Copy the folder and its contents
    shutil.copytree(source_folder, destination_folder)
    print(f"Folder copied to: {destination_folder}")
    
Note: for all NDC files not in English (i.e. Spanish or Arabic documents), were translated as txts into English using DeepL. This may not be the most accurate of mechanisms for translation, but given the limited timeframe given to us to complete this project, and the magnitude of datapoints available, it was felt that it would be the most precise tool given our project. 

Furthermore, we then added indicators to our data points, on our CSV file, including indicators related to region name, GDP per capita, HDI Index, etc. The sources for these datapoints are derived from the World Bank, UNSTATS and HDI).

*Here is a visual representation of our dataset*:
![Table with NDCs and Indicators](https://github.com/user-attachments/assets/f2476686-deb2-42f1-b3bc-b550e73ed4eb)

## Word Frequency

As a next step in our process, we felt it was necessary to determine what were our top words across all NDCs, but then also according to specific documents, in order to get a glimpse into the different NDCs, and their topics of interest.

In this next section, we will write out the code we used to find the top 25 words across all NDCs:

### Code:

    #Before starting, be sure to mount your drive if all of your txt files are on your drive!
    from google.colab import drive
    drive.mount('/content/drive')

    #Be sure to import the following (allows us to remove stopwords and access the words used across the NDCs)
    import os
    from collections import Counter
    from nltk.corpus import stopwords
    import string

    #Be sure to modify this according to your path folder!
    text_folder = "/content/drive/Shareddrives/NDC_txts"  # Make sure this path is correct
    stop_words = set(stopwords.words("english"))
    all_words = []

    #This allows us to filter out any punctuation and stop_words in our corpus:
    for filename in os.listdir(text_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(text_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().lower()
            text = text.translate(str.maketrans("", "", string.punctuation))
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
            all_words.extend(filtered_words)

    word_counts = Counter(all_words)
    top_25_words = word_counts.most_common(25)
    
    print("\nTop 25 words across all NDCs:")
    for word, count in top_25_words:
        print(f"  {word}: {count}")

With this code, we obtained the following results:

*Top 25 words across all NDCs:*

| Top Word | Count |
| --- | --- |
|climate | 27905 |
  change | 17122
  national | 16078
  emissions | 14586
  energy | 12435
  ndc | 11813
  adaptation | 11524
  development | 11393
  sector | 10952
  mitigation | 8531
  contribution | 7352
  implementation | 7293
  management | 6965
  water | 6880
  measures | 6617
  determined | 6199
  ghg | 6023
  nationally | 5603
  actions | 5493
  sectors | 5463
  including | 5451
  sustainable | 5407
  use | 5266
  reduction | 5183
  information | 5124

These words will inspire us when it comes to choosing the key words we will pursue in greater depth throughout our analysis.

For reference, the following is the code used in order to determine the top 10 words used for each NDC.

### Code:
    import os
    import string
    from collections import Counter
    from nltk.corpus import stopwords
    
    text_folder = "/content/drive/Shareddrives/NDC_txts"
    stop_words = set(stopwords.words("english"))
    word_freq_by_file = {}
    
    # Loop through each text file
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(text_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().lower()
    
                # Remove punctuation and split into words
                text = text.translate(str.maketrans("", "", string.punctuation))
                words = text.split()
    
                # Remove stopwords
                filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    
                # Count word frequencies
                word_counts = Counter(filtered_words)
    
                # Save top 10 words
                word_freq_by_file[filename] = word_counts.most_common(10)
    
    # Display results
    for file, top_words in word_freq_by_file.items():
        print(f"\n📄 Top words in {file}:")
        for word, count in top_words:
            print(f"   {word}: {count}")

*If you have the chance, we recommend taking a glance at the results that appear!* For instance, just so you may have a glance, here are some examples of the top words of the NDCs of Cambodia, Canada, Indonesia and Turkmenistan.

![Cambodia](https://github.com/user-attachments/assets/97fd765c-c905-4390-90e1-4ac850ea668f)
![Canada](https://github.com/user-attachments/assets/4ba75e2a-543c-4df1-a5e8-76256b4556b5)
![Indonesia](https://github.com/user-attachments/assets/8ac76ed4-89d6-4177-83cb-f47c00355b62)
![Turkmenistan](https://github.com/user-attachments/assets/8c112bc2-bb67-47dc-b693-84842a43da53)

## Topic Modeling

## Counting Words

### Method

We created a folder of NDC .txt files and a .csv file with a "filename" column containing the title of each .txt file in the folder, indexed to country codes. Next, we loaded the .csv file into a dataframe and defined a read_text function to read the .txt file corresponding to each title in the "filename" column of the .csv. The resulting dataframe includes a "text_content" column with the content of each NDC.

    load libraries
    import spacy
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MultiLabelBinarizer

    !python3 -m spacy download en_core_web_sm

    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # Path to the CSV and folder with text files
    txt_folder = "/content/drive/Shareddrives/NDC_txts"
    csv_path = "/content/drive/Shareddrives/NDC_txts/Reg_NDC_1.csv"
    
    # Loading the CSV
    df = pd.read_csv(csv_path)
    
    # Function to read text from file
    def read_text(filename):
        file_path = os.path.join(txt_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None  # or return "" if you prefer an empty string
    
    # Applying the function to the 'filename' column
    df["text_content"] = df["filename"].apply(read_text)

Next, we started counting words and phrases in the "text_content" column. For words, we defined the function normalize_word_count that counts the number of times a target word appears in each entry in the "text_content" column and normalizes that figure based on the length of the document ("doc_length") with stop words eliminated. We then created a new column, "normalized counts," for the normalized count of the target word for each NDC.

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
    
    def normalize_word_count(df, target_word, normalization):
        normalized_counts = []
        target_word = target_word.lower()
    
        for text in tqdm(df["text_content"]):
            doc_nlp = nlp(text)
    
            # Count how many times the word appears
            word_count = sum(1 for token in doc_nlp if token.text.lower() == target_word)
    
            if normalization == "doc_length":
                total = len(text)
    
            elif normalization == "word_count":
                total = sum(1 for token in doc_nlp if not token.is_punct and not token.is_space)
    
            elif normalization == "nonstopword_count":
                total = sum(1 for token in doc_nlp if not token.is_punct and not token.is_space and not token.is_stop)
    
            else:
                raise ValueError(f"Unknown normalization option: {normalization}")
    
            normalized_value = word_count / total if total > 0 else 0
            normalized_counts.append(normalized_value)
    
        # Creating the new column
        col_name = f"{target_word}_count_normalized"
        df[col_name] = normalized_counts
    
        print(f"Column '{col_name}' created successfully.")
        return df

        # Running the code
        df = normalize_word_count(df, target_word = '[word]', normalization = 'nonstopword_count')

        # Converting the dataframe back to a CSV
        df.to_csv('/content/drive/Shareddrives/NDC_txts/Reg_NDC_1.csv', index=False)

For phrases, the process is similar. The code below can handle multiple phrases at at time.

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
    
    def normalize_multi_phrase_count(df, phrases, normalization="word_count"):
        phrases = [phrase.lower() for phrase in phrases]
    
        phrase_counts = {phrase: [] for phrase in phrases}
    
        for text in tqdm(df["text_content"]):
            doc = nlp(text.lower())
            doc_text = doc.text
    
            if normalization == "doc_length":
                total = len(text)
    
            elif normalization == "word_count":
                total = sum(1 for token in doc if not token.is_punct and not token.is_space)
    
            elif normalization == "nonstopword_count":
                total = sum(1 for token in doc if not token.is_punct and not token.is_space and not token.is_stop)
    
            else:
                raise ValueError(f"Unknown normalization option: {normalization}")
    
            for phrase in phrases:
              raw_count = doc.text.count(phrase)
              normalized_count = raw_count / total if total > 0 else 0
              phrase_counts[phrase].append(normalized_count)
    
        # Creating the new column
        for phrase in phrases:
            col_name = f"{phrase.replace(' ', '_')}_count_normalized"
            df[col_name] = phrase_counts[phrase]
            print(f"Column '{col_name}' created successfully.")
    
        return df

        # Defining the phrases and running the code
        phrases = ["[phrase 1]", "[phrase 2]", "[phrase 3]"]
        df = normalize_multi_phrase_count(df, phrases, normalization = 'nonstopword_count')

        # Converting the dataframe back to a CSV
        df.to_csv('/content/drive/Shareddrives/NDC_txts/Reg_NDC_1.csv', index=False)

We subsequently prepared two visualizations: bar charts, disagreggated by indicators such as income level and sub-region, and cloropleth maps, which use the ISO 3166-1 alpha 3 country codes. 

For a single word, we generated a bar chart by grouping the average normalized word count for a given word by the selected indicator and plotting it.

        import matplotlib.pyplot as plt
        
        # Assuming '[Indicator]' is a column in the DataFrame 'df', and we want to plot the '[word]_count' by '[Indicator]':
        
        # Group the DataFrame by '[Indicator]' and calculate the mean growth count for each [indicator]
        [Indicator]_[word]_counts = df.groupby('[Indicator]')['[word]_count_normalized'].mean()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        [Indicator]_[word]_counts.plot(kind='bar')
        plt.title('Average ["Word"] Count by [Indicator]')
        plt.xlabel('[Indicator]')
        plt.ylabel('Average ["Word"] Count')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()

To create a chloropleth map, we used the following code: 

        import plotly.express as px
        
        fig = px.choropleth(
            df,
            locations="Code",                   # Name of the column with ISO 3166-1 alpha-3 codes
            color="[word]_count_normalized",    # Variable represented by color
            hover_name="Party",                 # Information displayed when hovering over each country
            color_continuous_scale="Viridis",   # Color Palette
            title="'Word' count by country",    # Title
            projection="natural earth"          # Cartographic projection style
        )
        
        fig.show()

For very low-frequency phrases, we also created a find_countries_with_phrase function to list the names of the countries whose NDCs include them.

        def find_countries_with_phrase(df, phrase):
            phrase = phrase.lower()  # Ensure the phrase is in lowercase
        
            countries_with_phrase = []
        
            # Iterate over the DataFrame rows
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                text = row["text_content"]
                country = row["Party"]  # Replace 'Country' with your actual column name for countries/parties
        
                # Process the text content
                doc = nlp(text.lower())
                doc_text = doc.text
        
                # Count occurrences of the phrase in the text
                raw_count = doc_text.count(phrase)
        
                # If the raw count is greater than 0, add the country to the list
                if raw_count > 0:
                    countries_with_phrase.append(country)
        
            return countries_with_phrase
        
        # Application:
        phrase = "planetary boundaries"
        countries = find_countries_with_phrase(df, phrase)
        
        # Display the list of countries
        print("Countries with 'planetary boundaries' in their NDCs:", countries)
        
### Theme 1 Visualizations: Post-Growth

The first cluster of visualizations demonstrate word counts for terms associated with **post-growth discourse** [EXPLAIN?], such as "steady state” and "degrowth," and the recognition of ecological limits, “planetary boundaries.”

#### "Steady State"
![Planetary Boundaries](https://github.com/user-attachments/assets/1839cf67-0b04-4e6d-915a-b1dfde501e42)

#### "Planetary Boundaries"
![Steady State](https://github.com/user-attachments/assets/8be981d6-c4aa-42e9-bb37-ad220c8afa75)

The phrase “steady state” is only used in Barbados’s NDC, and “planetary boundaries” in Cabo Verde’s and Liberia’s. The word “degrowth” appears in no NDCs in the corpus. [DISCUSS]

### Theme 2 Visualizations: Economic Growth and Development

Having demonstrated the lack of terms from discourses critical of economic growth in an ecological context, we tested  "economic growth" and "development," which revealed themselves to be far more prevalent across the corpus.

#### "Economic Growth"
![Economic Growth](https://github.com/user-attachments/assets/fd04b97d-634e-4eb9-b92e-abb4f1c824b6)

![Economic Growth - Sub-Region Graph](https://github.com/user-attachments/assets/c8846136-da99-425d-9ace-a005a9f50927)

#### "Development"
![development - map2](https://github.com/user-attachments/assets/06b14a73-eb84-4dec-9de7-25834faf0889)
![Development - Graph - subregion](https://github.com/user-attachments/assets/56415471-bc27-4f4e-8123-184b9f58ed3d)
![Development - Graph - income](https://github.com/user-attachments/assets/104476c6-b3c3-443c-b856-6f0ab4dba236)

### Visualizations: The Private Sector

Relatedly, the next group of words have to do with the prevalence of "innovation," "technology," and the "private" sector in the NDCs.

#### "Innovation"
![innovation - map](https://github.com/user-attachments/assets/6ead6998-8c93-4b8a-a5b7-39a34a4e4703)
![innovation - graph - subregion](https://github.com/user-attachments/assets/77b9f22e-6a6e-4f70-9b7c-40d4f94517a9)
![innovation - graph - income](https://github.com/user-attachments/assets/4906f51d-48fe-4109-9a88-3e47676c0794)

#### "Technology"
![technology - map](https://github.com/user-attachments/assets/a3c21c8a-bcd8-40cb-9561-dbbf0f683f37)
![technology - graph - subregion](https://github.com/user-attachments/assets/99f1fc72-b63d-4ac5-8d1d-c40be9473a14)

#### "Private"
![private - map](https://github.com/user-attachments/assets/73fc9aa8-b167-4e41-9025-7835e4c6d451)
![private - graph - subregion](https://github.com/user-attachments/assets/76ec638e-7d84-40ba-8493-07a4831c6f96)
![private - graph - income level](https://github.com/user-attachments/assets/4fb6d827-4338-4201-9ff0-6296931e6d64)

### Visualizations: Limits

We considered the idea of limitations, such as the principle of the "sustainable" and the role of "regulation."

#### "Regulation
![regulation - map](https://github.com/user-attachments/assets/237bb907-3db6-4731-be64-adb786ef7b1e)
![regulation - graph - subregion](https://github.com/user-attachments/assets/29ad570a-7ba7-4553-942a-a399ba21af95)

#### "Sustainable"
![Sustainable](https://github.com/user-attachments/assets/c1c409b0-e003-4f87-b527-f6989784428b)

![Sustainable - Income Level Graph](https://github.com/user-attachments/assets/84e5d6a3-81d4-4a92-8bee-5d9d106d8685)
![Sustainable - Sub-Region Graph](https://github.com/user-attachments/assets/65369869-c3e7-4ee9-9881-f470c7943f55)

### Visualizations: Energy

We tested a set of words associated with **energy** sources, including "fossil fuels," "energy transition," and "clean energy."

##### "Fossil Fuels"
![Fossil Fuels](https://github.com/user-attachments/assets/d5f2cf12-e6b4-40ba-8f5a-3776ba6a42cf)

![Fossil Fuels - Graph](https://github.com/user-attachments/assets/ced0a6e3-4b07-4f0b-b716-11f38c0c000d)
![Fossil Fuels - Sub-Region Graph](https://github.com/user-attachments/assets/d78578d4-d341-4e3f-8060-858ec7f362e0)

#### "Energy Transition"
![Energy Transition](https://github.com/user-attachments/assets/37d9e460-55aa-45db-a450-ca2d28f9f1b3)

![Energy Transition - Income Level Graph](https://github.com/user-attachments/assets/2a55b3af-4328-43a6-a859-dfec6528a8dd)
![Energy Transition - Sub-Region Graph](https://github.com/user-attachments/assets/849745ba-4c6e-4bd5-8ebb-e04ca6e422ea)

#### "Clean Energy"
![Clean Energy](https://github.com/user-attachments/assets/7701a25b-9de5-40c1-b563-9b7b54c8c049)

![Clean Energy - Sub-Region Graph](https://github.com/user-attachments/assets/65a0649c-2f42-46d5-982e-063b4dd09a61)
![Clean Energy - Income Level Graph](https://github.com/user-attachments/assets/bf8b916f-139c-4035-ac4e-193d18320c73)

### Visualizations: Prevention vs. Disaster Management

We finally considered the prevalence of ideas like "mitigation," or reducing climate impacts, and "adaptation" to the effects of climate change, such as "disaster."

#### "Mitigation"
![Mitigation](https://github.com/user-attachments/assets/ec285821-07df-46b6-8ab6-f6b5af687fcd)

![Mitigation - Sub-Region Graph](https://github.com/user-attachments/assets/0036265e-409f-48a7-a12f-0c88fe2f6988)
![Mitigation - Income Level Graph](https://github.com/user-attachments/assets/a8933767-579a-442a-9169-5c2124a6d39f)

#### "Adaptation"
![Adaptation](https://github.com/user-attachments/assets/b9310e27-418a-4a53-baaa-9b9b69dbc52e)

![Adaptation - Sub-Region Graph](https://github.com/user-attachments/assets/8655cf6d-bc4d-41f0-b26c-335030eebfae)
![Adaptation - Income Level Graph](https://github.com/user-attachments/assets/681b7db4-5052-4a8f-8d22-3c83f4b6b522)

#### "Disaster"
![Disaster](https://github.com/user-attachments/assets/d180ad76-39a0-4e8c-8746-033d364f6be6)

![Disaster - Sub Region Graph](https://github.com/user-attachments/assets/3134cda9-c48e-4739-9ab3-b9b04ffc0483)
![Disaster - Income Level Graph](https://github.com/user-attachments/assets/3d5550f5-6a71-4937-9021-f11e11b1ca80)

### Regression Analysis

## Discussion


### Regression Analysis

## Discussion

### Limitations

Our data is quite sparse -> tried to take into account the length of the NDC files.
For Europe, all EU countries have the same NDC -> which might modify the data, we decided it would be pertinent to replicate it, which might make it difficult (word choice in these NDCs might thus be exaggerated), we tried to factor this in by focusing on sub-regions as point of analysis.

## Conclusion
