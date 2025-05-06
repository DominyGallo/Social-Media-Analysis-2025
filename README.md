# A Linguistic Map of Nationally Determined Contributions
Final project for Jean-Philippe COINTET's Social Media Analysis course by Dominy GALLO, Chiara HAMPTON, Maëlle LEFEUVRE, Camille LEFEVRE, and Chloe PRYCE.

Date: May 8, 2025

## Introduction

GRAPH STYLING: 

Title: Average Normalized Count of "[Word]” by Income Level/Sub-region etc
x-label: Average "[Word]" Count

MAP STYLING

Average Normalized Count of '[Word]' by Country


## Data Processing

## Results

### Word Frequency

### Topic Modeling

### Counting Words

#### Method

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
    csv_path = "/content/drive/Shareddrives/NDC_txts/ndcs - sorted with indicators - All indicators (1) (1).csv"
    
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

Next, we started counting words and phrases in the "text_content" column. For words, we defined the function normalize_word_count that counts the number of times a target word appears in each entry in the "text_content" column and normalizes that figure based on the length of the document ["doc_length"] with stop words eliminated. We then create a new column, "normalized counts," for the normalized count of the target word for each NDC.

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

For phrases, the process is similar.

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

Next, we prepared two visualizations: bar charts, disagreggated by indicators such as income level and sub-region, and cloropleth maps, which use the country codes. 

For a single word, we create a bar chart by grouping the average normalized word count for a given word by the selected indicator—in this case, sub-region—and plotting it.

        import matplotlib.pyplot as plt
        
        # Assuming 'Sub-region Name' is a column in the DataFrame 'df', and we want to plot the 'growth_count' by 'Party':
        
        # Group the DataFrame by 'Sub-region Name' and calculate the mean growth count for each party
        party_[word]_counts = df.groupby('Sub-region Name')['[word]_count_normalized'].mean()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        party_[word]_counts.plot(kind='bar')
        plt.title('Average ["Word"] Count per Sub-region')
        plt.xlabel('Sub-region Name')
        plt.ylabel('Average ["Word"] Count')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()

To create a chloropleth map, we used the following code: 

        import plotly.express as px
        
        fig = px.choropleth(
            df,
            locations="Code",                   # Name of the column with ISO alpha-3 codes
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
        
### Visualizations

#### Post-Growth Discourse

###### "Steady-State"
![Steady State - Map](https://github.com/user-attachments/assets/75133fe4-b9be-4c08-bca3-c01c5c889faa)
![Steady State - Sub-Region Graph](https://github.com/user-attachments/assets/7a36c452-cabc-4172-8e0c-20c2ca950f39)


##### "Planetary Boundaries"
![Planetary Boundaries - Map](https://github.com/user-attachments/assets/efa4f6af-5722-410d-afe6-bdd6b0e83d44)
![Planetary Boundaries - Graph](https://github.com/user-attachments/assets/9783ae93-aa73-43aa-9523-6976b968ce73)
![Planetary Boundaries - Sub-Region Graph](https://github.com/user-attachments/assets/4323ff8c-aba6-4927-9ff4-2254d169494c)

We found that the phrase "planetary boundaries" is only used in Cabo Verde and Liberia. 

##### "Degrowth", "Consumption" ?

#### Traditional economics discourse (?)

##### "Economic Growth"

##### "Green Growth"

##### "Sustainable"

#### Energy Themes

###### "Fossil Fuels"
![Fossil Fuels - Map](https://github.com/user-attachments/assets/9624682a-2359-4c23-b3dd-3c4a5fbe27a8)
![Fossil Fuels - Graph](https://github.com/user-attachments/assets/ced0a6e3-4b07-4f0b-b716-11f38c0c000d)
![Fossil Fuels - Sub-Region Graph](https://github.com/user-attachments/assets/d78578d4-d341-4e3f-8060-858ec7f362e0)

##### "Energy Transition"
![Energy Transition](https://github.com/user-attachments/assets/a2dfcb6d-97ae-43a3-b56a-6eef17b76d6c)

##### "Clean Energy"
![Clean Energy](https://github.com/user-attachments/assets/931fddca-b32c-411d-9614-efc3a2b4fa0a)

#### Policy Discourse (?)
##### "Mitigation"
![Mitigation](https://github.com/user-attachments/assets/43bc7340-b7d7-400c-9787-05abafc2b096)
![Mitigation - Sub-Region Graph](https://github.com/user-attachments/assets/0036265e-409f-48a7-a12f-0c88fe2f6988)
![Mitigation - Income Level Graph](https://github.com/user-attachments/assets/a8933767-579a-442a-9169-5c2124a6d39f)

##### "Adaptation"
![Adaptation](https://github.com/user-attachments/assets/86e43e64-a740-4947-8753-c015122e7481)
![Adaptation - Sub-Region Graph](https://github.com/user-attachments/assets/8655cf6d-bc4d-41f0-b26c-335030eebfae)
![Adaptation - Income Level Graph](https://github.com/user-attachments/assets/681b7db4-5052-4a8f-8d22-3c83f4b6b522)

##### "Disaster"

### Regression Analysis

## Discussion

## Conclusion


### Regression Analysis

## Discussion

## Conclusion
