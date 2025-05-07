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

For the purposes of this project, we have used the Nationally Determined Contributions (NDCs) provided in the UNFCCC Secretariat's database as a point of analysis. From this, we have approximately 200 datapoints, ranging from various countries and regions from around the world. Website: [https://unfccc.int/NDCRE] 

To obtain a base list of these documents along with their urls and essential metadata, we used the openclimatedata Github page ‘ndcs’: [https://github.com/openclimatedata/ndcs/blob/main/data/ndcs.csv]. This page provides an automatically updated record from the UNFCCC website in CSV form and includes country code, party (i.e. author country), title, type of document (NDC, addendum, translation, etc), language, URL, version and status (i.e. archived, active). 

As not all countries have provided multiple rounds of NDCs, we selected the most recent active NDC for each country and removed all others from the dataset. Some countries provided annexes or addendums introducing their NDCs; in these cases, we removed these supplementary documents as well to maintain one datapoint per country. 

To this refined CSV file, we added further indicators for each country. GDP data in USD was obtained from the World Bank: [https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?view=chart]. From UN statistics, we obtained region, sub-region and designation as Least Developed Countries (LDC), Small Island Developing States (SIDS), or neither. As LDC or SIDS designation was noted as a tick box in the UN data, we coded 1 for yes and 0 for no in our CSV. The same was done for OECD membership by cross referencing the members list from their website: [https://www.oecd.org/en.html]. Data was also taken from the UNDP Human Development Index: [https://hdr.undp.org/data-center/human-development-index#/indicies/HDI]. In Google Sheets, data from these various datasets were added in separate tabs alongside the ndcs Github download. Then, due to the common use of ISO 3-letter country codes by the UNFCCC, UN Stats, and World Bank across the datasets, it was possible to use VLOOKUP functions to add columns for these indicators in the master sheet. The HDI did not include these codes, therefore a VLOOKUP was done on the country name; any formula errors due to differences in country name spelling or abbreviation were corrected manually. 

The next step was to gather  the documents in a way where analysis could be made, through either word frequency, topic modelling, counting words or experimenting with regressions. Thus, to allow our readers to potentially replicate this, we have included here our process and code to obtain readable txt files from the NDC URLs:

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
    
In order to maintain the connection between the indicators CSV and the files in ndc_txts, we manually added a column for each txt file name in the format [Country].txt. This work was divided amongst the team members. 

Note: all NDC files not in English (i.e. Spanish, French, or Arabic documents) were translated as txts into English using DeepL. This may not be the most accurate of mechanisms for translation, but given the limited timeframe given to us to complete this project, and the magnitude of datapoints available, it was felt that it would be the most precise tool given our project. Please see the annex to this page for a discussion of how we checked for translation accuracy. 

*Here is a visual representation of our dataset*:
![Table with NDCs and Indicators](https://github.com/user-attachments/assets/f2476686-deb2-42f1-b3bc-b550e73ed4eb)

## Word Frequency

As a next step in our process, we felt it was necessary to determine what were our top words across all NDCs, but then also according to specific documents, in order to get a glimpse into the different NDCs, and their topics of interest.

In this next section, we will write out the code we used to find the top 25 words across all NDCs:

### Code to calculate the top 25 words across all NDCs:

Before starting, be sure to mount your drive if all of your txt files are on your drive!

    from google.colab import drive
    drive.mount('/content/drive')

Be sure to import the following (allows us to remove stopwords and access the words used across the NDCs)

    import os
    import nltk
    nltk.download("stopwords")
    from collections import Counter
    from nltk.corpus import stopwords
    import string

Be sure to modify this according to your path folder!

    text_folder = "/content/drive/Shareddrives/NDC_txts"  # Make sure this path is correct
    stop_words = set(stopwords.words("english"))
    all_words = []

This allows us to filter out any punctuation and stop_words in our corpus:

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

| Top 25 Words | Counts across NDCs |
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

### Code to calculate the top 10 words for each NDC:
    import os
    import string
    from collections import Counter
    from nltk.corpus import stopwords
    
    text_folder = "/content/drive/Shareddrives/NDC_txts"
    stop_words = set(stopwords.words("english"))
    word_freq_by_file = {}
    
 Then, loop through each text file:
    
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
    
Display results:
    
    for file, top_words in word_freq_by_file.items():
        print(f"\n📄 Top words in {file}:")
        for word, count in top_words:
            print(f"   {word}: {count}")

*If you have the chance, we recommend taking a glance at the results that appear!* For instance, just so you may have an idea, here are some examples of the top words of the NDCs of Cambodia, Canada, Indonesia and Turkmenistan.

![Cambodia](https://github.com/user-attachments/assets/7593d7ac-0390-47e0-a24e-6d7d5bab0c6d)
![Canada](https://github.com/user-attachments/assets/4ba75e2a-543c-4df1-a5e8-76256b4556b5)
![Indonesia](https://github.com/user-attachments/assets/792a3c53-3207-4142-87c1-86a9de6efeb8)
![Turkmenistan](https://github.com/user-attachments/assets/8c112bc2-bb67-47dc-b693-84842a43da53)

#### Analysis
- As can be seen, interestingly, Cambodia is one of the few NDCs where the terms "gender" and "women" is in the top 10 words used. It's important to note that these words are even used more often than the name of the country itself!
- When it comes to Canada, it seems pertinent that reference to "indigenous" people and most likely "first Nations" is in the top words used. It shows how the specific political context of states can be reflected through their priorities in their NDCs.
- In the case of Indonesia, it is interesting that "development" and "ecosystem" are in the top words used. While it is not surprising, it does suggest how important it is for Indonesia to 'adapt' to 'climate change', and to 'implement' 'change' as needed.
- Last but not least, Turkmenistan's top words are related to resources, with reference to "energy", "gas" and "water". This can perhaps shed some light as to what are the priorities of this country when it comes to framing its NDC.

## Topic Modeling

For topic modeling, we aimed to see what were the general topics that were formed across NDCs, by using BERTopic modeling. As the language employed by NDCs can be quite similar, the parameters for the BERTopic used had to be modified and played with in order to get pertinent results.

### Code:
    # load libraries
    import os
    import spacy
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MultiLabelBinarizer
       
You know the drill, mount your drive!
    
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

Download BERTopic
   
    !pip install bertopic
    !pip install sentence_transformers
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

After this script runs, documents will be a list of strings, where each string is the full content of one .txt file in the folder.
        
    text_folder = "/content/drive/Shareddrives/NDC_txts"
    documents = []  # Initialize an empty list to store your documents
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(text_folder, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())  # Append the content of each file to the list

Now 'documents' contains all the text from the .txt files. You can then proceed to generate the document_vectors using your model:
    
    # Ensure embeddings are generated from the 'documents' list
    model = SentenceTransformer("all-MiniLM-L6-v2")
    document_vectors = model.encode(documents, show_progress_bar=True)
    
    # Define UMAP, HDBSCAN, and vectorizer before using them in BERTopic
    umap_model = UMAP(n_neighbors=5, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
    umap_embeddings = umap_model.fit_transform(document_vectors)
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
    vectorizer = CountVectorizer(stop_words='english')
    
    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=True
    ).fit(documents, document_vectors)

In this next step, you can obtain the topic information (and check that the process actually worked!).

    topic_info = topic_model.get_topic_info()
    topic_info

Visualise as a barchart the topic modeling:

    topic_model.visualize_barchart(top_n_topics=22,n_words=10)

### Output
![Topic Modeling with BERTopic](https://github.com/user-attachments/assets/6b7b9d34-41c3-4bcf-a1c5-6c66d94d9839)


### Analysis

#### Limitations
As a next step, to ensure the reliability of results, it would be necessary to include code that would divide the txt files by paragraph. This would guarantee more accurate results than the ones we have obtained above. For the purposes of this project, the reliability of our topics is not concerning, as this modeling was used to inspire our selection of key words, alongside our top words across NDCs, rather than shaping our entire analysis. If you ever do manage to complete this step, we would be eager to see your results!


## Counting Words

### Literature Review

What words? Eco-social policy typology & themes in NDCs

perhaps @Chloe can do this?

--> perhaps help explaining which words we read --> based on the literature review done

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
        
### Theme 1 Visualizations: Post-Growth Discourses

The first cluster of visualizations demonstrate word counts for terms associated with **post-growth discourse**, such as "steady state” and "degrowth," and the recognition of ecological limits, “planetary boundaries.”

#### "Steady State"
![Planetary Boundaries](https://github.com/user-attachments/assets/1839cf67-0b04-4e6d-915a-b1dfde501e42)

#### "Planetary Boundaries"
![Steady State](https://github.com/user-attachments/assets/8be981d6-c4aa-42e9-bb37-ad220c8afa75)

The phrase “steady state” is only used in Barbados’s NDC, and “planetary boundaries” in Cabo Verde’s and Liberia’s. The word “degrowth” appears in no NDCs in the corpus. We can therefore conclude that post-growth and degrowth concepts, although circulating in certain spheres of academic discourse, have not been widely integrated into national emissions-reduction commitments.

### Theme 2 Visualizations: Economics

Having demonstrated the lack of terms from discourses critical of economic growth in an ecological context, we tested "economic growth" and "development," to test for persistent country interests in the relationship between emissions reduction and increasing GDP. These terms revealed themselves to be far more prevalent across the corpus. In this category, we also tested for common but "differentiated responsibilities," which accounts for the principle that higher-income states should contribute more financially to climate mitigation and adaptation efforts.

#### "Economic Growth"
![Economic Growth](https://github.com/user-attachments/assets/fd04b97d-634e-4eb9-b92e-abb4f1c824b6)
![Economic Growth - Sub-Region Graph](https://github.com/user-attachments/assets/c8846136-da99-425d-9ace-a005a9f50927)

As visible in the sub-region graph, “economic growth” appears most commonly in North American NDCs; from the map, we can observe that this is primarily accounted for by the United States. The phrase is least prevalent in Northern Europe. It is interesting that the three clusterings of Pacific island states have such different counts for the phrase: Polynesia has the second-lowest count and Micronesia the second-highest of all NDCs, with Melanesia in the middle.  On the map, we observe that Brazil, Saudi Arabia, some sub-saharan African states, and Australia include some degree of “economic growth” discourse, but there are not clearly discernible broad regional patterns. In general, however, it is relatively infrequent in comparison with other terms, explored below.


#### "Development"
![development - map2](https://github.com/user-attachments/assets/06b14a73-eb84-4dec-9de7-25834faf0889)
![Development - Graph - subregion](https://github.com/user-attachments/assets/56415471-bc27-4f4e-8123-184b9f58ed3d)
![Development - Graph - income](https://github.com/user-attachments/assets/104476c6-b3c3-443c-b856-6f0ab4dba236)

“Development” yields a much clearer picture. It is least prevalent in Northern and Western Europe and Australia and New Zealand, then Northern America and Eastern Europe. We should note the significant difference between the United States and Canada’s NDCs and Mexico’s, which has a much higher prevalence for the word, as indicated on the map. We can thus observe that those regions commonly referred to as the “Global North”—Europe, the U.S. and Canada, and Australia and New Zealand—are least concerned with “development” in their emissions-reduction commitments.

By contrast, the term is most prevalent in Asia, in particular Central Asia, then South-eastern and Southern Asia. Per the map, China’s normalized count is relatively high, alongside several Central Asian states: Turkmenistan, Uzbekistan, Afghanistan, Tajikistan, and Kyrgistan. Thailand, the Philippines, Indonesia, and especially Papua New Guinea, moving east into the Pacific, also have notably high counts. Russia and Brazil, though lower on the scale than China, appear to have similarly high frequencies for the term. In Latin America, the frequency is consistently moderate to high. Among African States, frequencies are inconsistent though generally moderate, with Mauritania, Tanzania, and Somalia standing out for their particularly high counts.

The Income Level graph reveals, indeed, that high income states are the least likely to mention development in their NDCs, a phenomenon reflected in the relative scarcity of the term in North America, Europe, and Australia. From low to upper-middle income, however, income level seems not to be strongly correlated to prevalence and regional trends appear to dominate, with development discourse most prevalent in Asia and the Pacific, then in Latin America and Africa, and least of all in the “Global North.”

#### "Differentiated Responsibilities"
![responsibilities](https://github.com/user-attachments/assets/42361e67-ad20-48fa-ad37-d0fe86be1938)
![responsibilities - subregion](https://github.com/user-attachments/assets/9d0eadb2-951e-40fe-9b02-2ebd878dba8f)
![responsibilities - income](https://github.com/user-attachments/assets/cee0953d-040a-43bb-a6ca-0dbcd617e289)

Notably, states likely to incur the most obligations based on the principle of “common but differentiated responsibilities” make the least reference to the phrase—it appears nowhere in any Western or Eastern European, Northern American, or Australian NDC. It appears most frequently in Western Asia and Latin America and the Caribbean, with particularly high rates in Brazil. The highest and the lowest income states, interestingly, are the least likely to mention it in comparison with middle-income states.


### Theme 3 Visualizations: Innovation, the Private Sector, and the State

The next group of words considers the NDCs’ ideological affiliation with “innovation” and “technology” as tools to address the climate crisis. We have paired this set of discourses with the role of the “private” sector and, by contrast, the state's role in "regulation."

#### "Innovation"
![innovation - map](https://github.com/user-attachments/assets/6ead6998-8c93-4b8a-a5b7-39a34a4e4703)
![innovation - graph - subregion](https://github.com/user-attachments/assets/77b9f22e-6a6e-4f70-9b7c-40d4f94517a9)
![innovation - graph - income](https://github.com/user-attachments/assets/4906f51d-48fe-4109-9a88-3e47676c0794)

We can observe that “innovation” is far and away most prevalent in Northern American NDCs, especially in the United States and Mexico. It is notably prevalent in Saudi Arabia’s NDC’s, China’s, Canada’s, and Brazil’s, but these outliers do not contribute to consistent regional significance anywhere but in Northern America, per the sub-region graph. Western Europe has the second-highest prevalence of the term, though it is significantly behind North America in this regard. The Income Level graph demonstrates a strong correlation between high income and high prevalence of “innovation” in NDCs—the lower the income, the less frequent is the term. 

#### "Technology"
![technology - map](https://github.com/user-attachments/assets/a3c21c8a-bcd8-40cb-9561-dbbf0f683f37)
![technology - graph - subregion](https://github.com/user-attachments/assets/99f1fc72-b63d-4ac5-8d1d-c40be9473a14)

“Technology” shows a slightly different picture. It is most prevalent in Southern and South-Eastern Asia, with notable frequency in India and some prevalence in Saudi Arabia, Iraq, Afghanistan, Indonesia, and Thailand. It is also relatively frequent in China. It appears almost not at all anywhere in Europe. 

It is not clear, from a simple word-counting process, to determine the context in which “technology” is used. It could indicate the belief in technology as a solution to the climate crisis. The lack of overlap between the technology and innovation graphs, could, however, indicate that “technology” acquires different meanings. It could be used in the context of actual technology transfers, or low-technology states requesting technological assistance from higher-tech states, in contrast to the principle of technological innovation.

#### "Private"
![private - map](https://github.com/user-attachments/assets/73fc9aa8-b167-4e41-9025-7835e4c6d451)
![private - graph - subregion](https://github.com/user-attachments/assets/76ec638e-7d84-40ba-8493-07a4831c6f96)
![private - graph - income level](https://github.com/user-attachments/assets/4fb6d827-4338-4201-9ff0-6296931e6d64)

The “private” graph presents the striking prevalence of private-sector solutions coming out of the United States, which is individually  responsible for the high prevalence of the term in North American NDC’s. It appears, by contrast, almost nowhere in Europe or the major Asian states, Russia, China, and India. It scarcely appears, likewise, in Australia or Canada. It has some prevalence in Central Asian and Middle Eastern states, some sub-Saharan African states, some South-Asian and Pacific states, and Brazil, but nowhere is its prevalence comparable to that of the United States.

Interestingly, in spite of the over-representation of the term in the high-GDP state of the US, the relative lack of prevalence in other high-GDP states (such as European states and China) reveal no correlation between income and prevalence of “private” sector ideas. Reliance on the private sector appears, that is, more a matter of cultural than purely economic traits.

#### "Regulation"
![regulation - map](https://github.com/user-attachments/assets/237bb907-3db6-4731-be64-adb786ef7b1e)
![regulation - graph - subregion](https://github.com/user-attachments/assets/29ad570a-7ba7-4553-942a-a399ba21af95)

This map indicates that regulation is prevalent almost exclusively in European NDCs. It appears almost nowhere else.

### Theme 4 Visualizations: Energy

We then tested a set of words associated with **energy** policy, including "fossil fuels," "energy transition," and "clean energy."

#### "Fossil Fuels"
![Fossil Fuels](https://github.com/user-attachments/assets/d5f2cf12-e6b4-40ba-8f5a-3776ba6a42cf)
![Fossil Fuels - Graph](https://github.com/user-attachments/assets/ced0a6e3-4b07-4f0b-b716-11f38c0c000d)

The phrase “fossil fuels” is most prevalent in China, Afghanistan, and Brazil. It appears almost not at all in Sub-Saharan Africa, all of the Pacific states between India and Australia, Russia, Mexico, and Western Latin American states. It has some prevalence, by contrast, in the United States and Europe. On the Income Level graph, we observe indeed that high income countries are most likely to discuss fossil fuels in their NDCs, and low and lower-middle income states are significantly  less likely to do so. 

#### "Energy Transition"
![Energy Transition](https://github.com/user-attachments/assets/37d9e460-55aa-45db-a450-ca2d28f9f1b3)
![Energy Transition - Income Level Graph](https://github.com/user-attachments/assets/2a55b3af-4328-43a6-a859-dfec6528a8dd)
![Energy Transition - Sub-Region Graph](https://github.com/user-attachments/assets/849745ba-4c6e-4bd5-8ebb-e04ca6e422ea)

Brazil, the United Kingdom, and Cuba appear to have the highest prevalence of the phrase “energy transition” by far; by contrast, the phrase appears almost nowhere in Europe, Asia, the Middle East, Sub-Saharan Africa, or the rest of Latin America. It appears with some frequency in North America and Australia. The Income Level graph reveals that high and upper-middle income countries use this phrase on average more frequently than low and lower-middle income countries. When we examine by region, the average is surprising—Northern America has far and away the highest averaged normalized count, despite the fact that none of its states have as high frequency levels as Brazil and the UK, because rates are moderately high across the region rather than split between singular highs and many lows, as is the case in Latin America (Brazil’s region) and Europe (the UK’s).

#### "Clean Energy"
![Clean Energy](https://github.com/user-attachments/assets/7701a25b-9de5-40c1-b563-9b7b54c8c049)
![Clean Energy - Sub-Region Graph](https://github.com/user-attachments/assets/65a0649c-2f42-46d5-982e-063b4dd09a61)

The phrase “clean energy” is another term most prevalent by far in the United States, as confirmed on the regional graph. The North American States are seconded by Australia, but by a wide margin. The phrase appears almost nowhere else.

#### "Renewable"
![renewable](https://github.com/user-attachments/assets/3be13147-b0a3-4f6b-8823-f18e65c6982a)
![renewable - graph - subregion](https://github.com/user-attachments/assets/78e39c1d-53ba-4bbd-bdac-23d21a4b74e8)
![renewable - graph - income](https://github.com/user-attachments/assets/a95460c1-872b-46b4-9013-d1535ab46621)

“Renewable” occurs more frequently than “clean” as a denotation for “energy” across the board. It is least prevalent in Europe, Northern America, Eastern Asia, and Australia and New Zealand, and most prevalent in Polynesia, Micronesia, Melanesia, North Africa, and Western Asia, with a notably high prevalence in Saudi Arabia.  No correlation can be identified based on income.

### Theme 5 Visualizations: Mitigation v. Adaptation

We next considered the prevalence of ideas like "mitigation," or reducing climate impacts, and "adaptation" to the effects of climate change.

#### "Mitigation"
![Mitigation](https://github.com/user-attachments/assets/ec285821-07df-46b6-8ab6-f6b5af687fcd)
![Mitigation - Sub-Region Graph](https://github.com/user-attachments/assets/0036265e-409f-48a7-a12f-0c88fe2f6988)
![Mitigation - Income Level Graph](https://github.com/user-attachments/assets/a8933767-579a-442a-9169-5c2124a6d39f)

A cursory look at the graph reveals that “mitigation,” or prevention of climate change, is far more prevalent in the Southern Hemisphere than the Northern. Europe and Northern America have the lowest frequencies for the term. It is most prevalent in Sub-Saharan and Northern Africa, followed by South-eastern Asia and Latin America and the Caribbean. It is scarcely mentioned by the US, Canada, European States, Russia, or China. The Income Level graph indicates indeed that high income states have the fewest references to “mitigation,” and low-income states, the most.

#### "Adaptation"
![Adaptation](https://github.com/user-attachments/assets/b9310e27-418a-4a53-baaa-9b9b69dbc52e)
![Adaptation - Sub-Region Graph](https://github.com/user-attachments/assets/8655cf6d-bc4d-41f0-b26c-335030eebfae)
![Adaptation - Income Level Graph](https://github.com/user-attachments/assets/681b7db4-5052-4a8f-8d22-3c83f4b6b522)

“Adaptation” reveals a slightly different picture. It is similarly least prevalent in Europe, North America, and New Zealand, but Russia and China break from the “mitigation” trend and do make reference to “adaptation.” It is most prevalent in Melanesia, then Central Asia, followed by Northern Africa, Latin America and the Caribbean, South-eastern Asia, and finally Sub-Saharan Africa. Within Latin America, interestingly, Brazil includes rather few references while Argentina, Uruguay, Colombia, Ecuador, and Peru include more frequent notes.

### Theme 6 Visualizations: Land-Related Discourses

Finally, we look into terms that have more geographically specific relevance, such as "disaster," "agriculture," and "water."

#### "Disaster"
![Disaster](https://github.com/user-attachments/assets/d180ad76-39a0-4e8c-8746-033d364f6be6)
![Disaster - Sub Region Graph](https://github.com/user-attachments/assets/3134cda9-c48e-4739-9ab3-b9b04ffc0483)

A useful term to think about “adaptation” is “disaster,” as an index of references to climate impacts via natural disasters. Indeed, the results reveal an over-representation in the Pacific islands of Melanesia and Polynesia and South-eastern and Southern Asia. They appear scarcely at all in Europe—not at all in Western Europe—and minimally in Northern America or Australia and New Zealand. 

Although the regions with high “disaster” counts are notably exposed to natural disasters, infrequency of the word does not correlate to infrequency of hazards, as indicated by this map (source: https://journals.openedition.org/cybergeo/25297?lang=es): 

![Disaster Map](https://github.com/user-attachments/assets/45bec81a-065f-4316-8f22-567f5ed06d9b)

Moreover, high income states seem to have the lowest prevalence by far of the term, but the trend is not linear.

![Disaster - Income Level Graph](https://github.com/user-attachments/assets/3d5550f5-6a71-4937-9021-f11e11b1ca80)

#### "Agriculture"
![agriculture](https://github.com/user-attachments/assets/e0385802-90a2-487c-bda9-3a6bf945c926)
![agriculture - graph - subregion](https://github.com/user-attachments/assets/5ddaac88-de1c-4379-b343-4cc6e2439322)
![agriculture - graph - income](https://github.com/user-attachments/assets/35d695fd-48ed-4a2a-a728-95ed9a0f8342)

The word “agriculture” is most common in Central and Southern/Southeastern Asia and Sub-Saharan Africa. Observing the map, it appears to be a concern for many African states, with at least two outliers (Mali and Kenya) accounted for by lack of available data. By Income Level, we can observe that high income states are least likely to discuss agriculture and low-income states most likely.

#### "Water"
![water](https://github.com/user-attachments/assets/6672bbe5-cb4f-4043-aa28-296a66fd6490)

Finally, we observe a strip of land in Central Asia, the Middle East, and Africa for whom “water” is a primary concern. Arid states such as Jordan, Iraq, and Saudi Arabia demonstrate particularly high frequency for the term. The worldwide map does not exactly correlate to a map of water stress, although there is some overlap (source: World Resources Institute):

![23-08-02-aqueduct-4 0-launch-global-countries-baseline_Insights](https://github.com/user-attachments/assets/5b9aab85-bba7-41bb-9366-d5ea31e234ba)


### Regression Analysis

## Discussion

### Limitations

Our data is quite sparse -> tried to take into account the length of the NDC files.
For Europe, all EU countries have the same NDC -> which might modify the data, we decided it would be pertinent to replicate it, which might make it difficult (word choice in these NDCs might thus be exaggerated), we tried to factor this in by focusing on sub-regions as point of analysis.

## Conclusion
