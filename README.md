# A Linguistic Map of Nationally Determined Contributions
Final project for Jean-Philippe COINTET's Social Media Analysis course by Dominy GALLO, Chiara HAMPTON, Maëlle LEFEUVRE, Camille LEFEVRE, and Chloe PRYCE.

Date: May 8, 2025

## Introduction

In 2015, 195 countries signed the Paris Agreement during COP 21, committing to limit global warming below 2°C, and as close to 1,5°C, compared to pre-industrial time. The parties engaged themselves to strengthen climate resilience and ensure that sufficient financial flows would be allocated to reach these goals globally. All member countries thus needed to elaborate a Nationally Determined Contribution (NDC) detailing the ambitious efforts they would undertake in order to reduce their emissions and adapt to climate change. Each NDC is also adapted to national circumstances, for instance to take into account specific geographic vulnerability or development challenges that some countries may face. According to the Paris Agreement, parties needed to deliver the first version of their NDC in 2020. It then needs to be updated every 5 years to ensure that the countries will pursue their efforts in the long term and keep increasing their ambition in terms of emission reduction. 

Through its NDC, each country put forward its target sectors and its strategy for climate action and sustainable development revealing a specific development paradigm that may differ between countries. Even if countries are legally binded to craft an NDC, there is a relative freedom in how the NDC should be designed, as for example the variety of length or languages attests. Therefore, the way NDCs are organised and worded can reveal information about a country’s priorities and understanding of climate change.

Existing studies have applied natural language processing (NLP) techniques to the NDCs for various purposes. Given that NDCs can vary substantially in detail and length, researchers have noted that their lack of structure can pose ‘substantial challenges to aggregating and analyzing the data, thereby complicating the assessment of global and national efforts in addressing climate change’ (Singh et al., 2024, p.1). A common focus has therefore been on using machine learning methods to annotate NDCs, either for specific discourses and commitments, or for evaluation against existing standards like the UN Sustainable Development Goals (Singh et al. 2024; Spokoyny et al. 2024). Seeking to address the issue of NDCs’ lack of transparency ‘not only about mitigation targets but also concerning implementation, financing and adaptation components’, Savin et al. (2025) used topic modelling on their NDC corpus to identify 21 topics, which were then grouped into 7 themes, such as development, implementation plans, and policies and technologies. This enabled them to analyze percentages of the NDCs devoted to different strategies within the target setting process. Interestingly, their findings revealed that ‘only ~20% of the total textual content in NDCs directly pertains to emission targets’ (ibid 303). This validates one of the premises of our research project: that examining the linguistic content of the NDCs can help provide insights beyond climate targets themselves into a country’s geopolitical positioning in the international climate debate or their advocacy for an ideological approach to addressing the climate crisis. 

Other studies have focused more on the relationship between the profiles of the parties to the UNFCCC and their policy discourses. Jernnäs and Linnér (2019) conducted a manual coding study of the NDCs to determine discourses and their correspondence with expected blocs in international governance. Despite looking at multiple specific discourses, they also examine whether these discourses broadly conform or reject ‘liberal environmentalism’, a framing in which ‘growth is perceived not only as compatible with but as a prerequisite for environmental protection’ (74). They conclude that the majority fall within the liberal environmentalism framework. Along with other studies cited here, they examine the possible effects of income, region and negotiation coalition on their results, suggesting that further research could expand to include more diverse indicators such as the Human Development Index  – we take this forward by considering this indicator, among others, in our analysis. 

With the 1,5°C Paris target seeming  increasingly out of reach, far  greater emissions cuts are needed by the countries most responsible, and wide implementation gaps exist between current and required funding streams from the Global North to the Global South for mitigation, adaptation and loss and damage. In this context, we perform a spatial analysis of language in the NDCs, primarily through word frequency. Although our initial interest was on mapping green growth vs post-growth discourses, our research process guided us towards a broader consideration of themes including energy, technology, natural resources, and beyond.  

## Data Processing

For the purposes of this project, we have used the Nationally Determined Contributions (NDCs) provided in the UNFCCC Secretariat's database as a point of analysis. From this, we have approximately 200 datapoints, ranging from various countries and regions from around the world. Website: [https://unfccc.int/NDCRE] 

To obtain a base list of these documents along with their urls and essential metadata, we used the openclimatedata Github page ‘ndcs’: [https://github.com/openclimatedata/ndcs/blob/main/data/ndcs.csv]. This page provides an automatically updated record from the UNFCCC website in CSV form and includes country code, party (i.e. author country), title, type of document (NDC, addendum, translation, etc), language, URL, version and status (i.e. archived, active). 

As not all countries have provided multiple rounds of NDCs, we selected the most recent active NDC for each country and removed all others from the dataset. Some countries provided annexes or addendums introducing their NDCs; in these cases, we removed these supplementary documents as well to maintain one datapoint per country. 

To this refined CSV file, we added further indicators for each country. GDP data in USD was obtained from the World Bank: [https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?view=chart]. From UN statistics, we obtained region, sub-region and designation as Least Developed Countries (LDC), Small Island Developing States (SIDS), or neither. As LDC or SIDS designation was noted as a tick box in the UN data, we coded 1 for yes and 0 for no in our CSV. The same was done for OECD membership by cross referencing the members list from their website: [https://www.oecd.org/en.html]. We also used human development scores from the UNDP Human Development Index: [https://hdr.undp.org/data-center/human-development-index#/indicies/HDI]. In Google Sheets, data from these various datasets were added in separate tabs alongside the ndcs Github download. Then, due to the common use of ISO 3-letter country codes by the UNFCCC, UN Stats, and World Bank across the datasets, it was possible to use VLOOKUP functions to add columns for these indicators in the master sheet. The HDI did not include these codes, therefore a VLOOKUP was done on the country name; any formula errors due to differences in country name spelling or abbreviation were corrected manually. 

The next step was to gather the documents in a way where we could conduct a language-processing analysis, through either word frequency, topic modelling, counting words or experimenting with regressions. Thus, to allow our readers to potentially replicate this, we have included here our process and code to obtain readable txt files from the NDC URLs:

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
    
In order to maintain the connection between the indicators (in the CSV spreadsheet) and the files in ndc_txts, we manually added a column for each txt file name in the format [Country].txt. This work was divided amongst the team members. As all European countries submitted an identifcal Europe-wide NDC text, we debated condensing them into a single EU datapoint but ultimately decided to keep the individual countries. This was to accurately reflect the number of countries aligned with an NDC discourse, even if they are part of a bloc. Furthermore, it enabled us to consider the relationship between the different indicators of European countries and their NDC text within our regressions and analysis of average word frequency per region, income, etc. 

Note:  all NDC files not in English (i.e. Spanish, French, or Arabic documents) were translated as txts into English using DeepL. This may not be the most accurate of mechanisms for translation, but given the limited timeframe given to us to complete this project, and the magnitude of datapoints available, it was felt that it would be the most precise tool given our project. Please see the annex to this page for a discussion of how we checked for translation accuracy. 

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

For reference, the following code was used to determine the top 10 words used for each NDC.

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

*If you have the chance, we recommend taking a glance at the results that appear!* For instance, just so you may have an idea, here are some examples of the top words of the NDCs of Cambodia, Canada, Indonesia and Turkmenistan (these results can also be found in our ipynb file titled "Word_Frequency.ipynb").

![Cambodia](https://github.com/user-attachments/assets/7593d7ac-0390-47e0-a24e-6d7d5bab0c6d)
![Canada](https://github.com/user-attachments/assets/4ba75e2a-543c-4df1-a5e8-76256b4556b5)
![Indonesia](https://github.com/user-attachments/assets/792a3c53-3207-4142-87c1-86a9de6efeb8)
![Turkmenistan](https://github.com/user-attachments/assets/8c112bc2-bb67-47dc-b693-84842a43da53)

#### Analysis
- As can be seen, interestingly, Cambodia is one of the few NDCs where the terms "gender" and "women" are in the top 10 words used. It's important to note that these words are even used more often than the name of the country itself!
- When it comes to Canada, it seems pertinent that reference to "indigenous" people and most likely "first Nations" is in the top words used. It shows how the specific political context of states can be reflected through their priorities in their NDCs.
- In the case of Indonesia, it is interesting that "development" and "ecosystem" are in the top words used. While it is not surprising, it does suggest how important it is for Indonesia to 'adapt' to 'climate change', and to 'implement' 'change' as needed.
- Last but not least, Turkmenistan's top words are related to resources, with reference to "energy", "gas" and "water". This can perhaps shed some light as to what are the priorities of this country when it comes to framing its NDC.

## Topic Modeling

For topic modeling, we aimed to see what were the general topics that were formed across NDCs, by using BERTopic modeling. As the language employed by NDCs can be quite similar, the parameters for the BERTopic model had to be modified and adjusted in order to obtain pertinent results.

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

    topic_model.visualize_barchart(top_n_topics=21,n_words=10)

### Topic Modeling Output
![Topic Modeling with BERTopic](https://github.com/user-attachments/assets/e0289141-38c6-419b-83e0-78a9affac5e3)



### Analysis
According to the parameters set through our BERTopic modelling, we obtained approximately 22 topics. As can be seen from a quick glance, the topics are quite various. Although many include the name of countries (and Topic 10 is not pertinent to our analysis), some insight can be obtained from these topics. For instance, topic 1 and topic 2 are quite similar when it comes to the words employed, and yet, there are still some differences. For instance, while both focus on "climate", the first topic includes key words related to "sector", "emissions" and "energy". When it comes to topic 2, there is instead reference to more nouns of action, such as "management", "adaptation", "contribution" or "mitigation".

*A few key observations*:
- **Topic 3**: Focusing on Oman, Egypt, Qatar and Kuwait, it is interesting that this is the few topics where the word "economic" is present, and in tandem with words such as "national", "sector", "water" and "energy".
- **Topic 12**: With reference to Korea, Georgia and Montenegro, the key topic words that appear include "carbon", "target", "emissions", "ghg", and most surprisingly "neutrality".
- **Topic 13**: It is quite staggering that the topic modelling was able to group together small island developing states together, such as Republic of Marshall Islands (RMI), the Solomon Islands and St Kitts and Nevis, with the words "change" and "adaptation" predominantely between them.
- **Topic 20**: in relation to our previous analysis of Cambodia individually, we see it present in Topic 20, alongisde Chile. Among all the topics, this is one the few that references to "women" and "gender". This thus sheds light as to what the priorities of these two countries is, relative to the other NDCs.

#### Limitations
As a next step, to ensure the reliability of results, it would be necessary to include code that would divide the txt files by paragraph. This would guarantee more accurate results than the ones we have obtained above. For the purposes of this project, the reliability of our topics is not concerning, as this modeling was used to inspire our selection of key words, alongside our top words across NDCs, rather than shaping our entire analysis. If you ever do manage to complete this step, we would be eager to see your results!


## Counting Words
The next step of our method involved counting specific words to understand countries' approach to meeting the Paris Agreement within their unique social and ecological context and culture. The actions by countries to reduce their national emissions can take on a multitude of dimensions, from the sectors they focus on to the balance of focus between technological solutions and behavioural solutions.  We conducted a short literature review to inform the words that we chose to count.

### Literature Review
We initially set out to understand countries' attitude towards economic growth in their NDCs. Given Vogel and Hickel's (2023) finding that the emissions reductions that high-income countries achieved through absolute decoupling fall far short of Paris-compliant rates, we were interested to understand the extent to which emissions reductions strategies centred around economic growth or "green growth".

To do so, we employed Schulze Waltrup's (2025) eco-social policy typology whereby the four types differ towards their approach towards economic growth. Based Schulze Waltrup's description of each type, we chose several words through which we could identify the type of policies which a countries' NDC focussed on. The typologies and associated words or phrases were as follows:

| Type of eco-social policy | Key words and phrases |
| --- | --- |
|Green growth | growth, economic growth, green growth |
  Recomposing Consumption and Production | consumption 
  Degrowth | planetary boundaries, steady state, degrowth

Given the unique ecological, social and economic complexities of each country, we were also interested in understanding the key themes or sectors countries' NDCs alluded to, and the extent to which the focus was on technological solutions. To determine the words for this, we referred to the EU Taxonomy Compass and the themes outlined in the NDC climate toolbox, from which we chose the following words:

- disaster
- energy
- sustainable
- mitigation
- adaptation
- fossil fuels
- clean energy
- energy transition
- development
- private
- technology
- innovation
- industry
- regulation
- differentiated
- renewable
- agriculture
- water


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
        
### Visualizations

In the section that follows, we group our word-frequency visualizations around themed clusters to investigate the ideological positioning and policy priorities of the states represented in the corpus. We begin with post-growth discourses (Theme 1) to test for the integration of a critique of economic growth into countries’ climate commitments. Theme 2 includes terms found in traditional economics, development, and foreign aid discourses. Our third theme clusters terms related to the balance between the private sector and the state and the role of innovation and technology in climate solution-building. Next, we consider energy-related discourses (Theme 4), from “fossil fuels” to the “renewable” and “clean.” Theme 5 balances two common terms in environmental policy discourses, mitigation and adaptation, to compare country priorities for prevention vs. disaster management, and Theme 6 clusters three terms related to geophysical properties, such as water and agriculture. Within each cluster, we consider differential word frequencies on the country level and seek to identify trends based on income level and sub-region.

### Theme 1: Post-Growth Discourses

The first cluster of visualizations demonstrate word counts for terms associated with **post-growth discourse**, such as "steady state” and "degrowth," and the recognition of ecological limits, “planetary boundaries.”

#### "Steady State"
![Planetary Boundaries](https://github.com/user-attachments/assets/1839cf67-0b04-4e6d-915a-b1dfde501e42)

#### "Planetary Boundaries"
![Steady State](https://github.com/user-attachments/assets/8be981d6-c4aa-42e9-bb37-ad220c8afa75)

The phrase “steady state” is only used in Barbados’s NDC, and “planetary boundaries” in Cabo Verde’s and Liberia’s. The word “degrowth” appears in no NDCs in the corpus. We can therefore conclude that post-growth and degrowth concepts, although circulating in certain spheres of academic discourse, have not been widely integrated into national emissions-reduction commitments.

### Theme 2: Green Growth

Having demonstrated the lack of terms from discourses critical of economic growth in an ecological context, we tested "economic growth" and "development," to test for persistent country interests in the relationship between emissions reduction and increasing GDP. These terms revealed themselves to be far more prevalent across the corpus. In this category, we also tested for common but "differentiated responsibilities," which accounts for the principle that higher-income states should contribute more financially to climate mitigation and adaptation efforts.

#### "Economic Growth"
![Economic Growth](https://github.com/user-attachments/assets/fd04b97d-634e-4eb9-b92e-abb4f1c824b6)
![Economic Growth - Sub-Region Graph](https://github.com/user-attachments/assets/c8846136-da99-425d-9ace-a005a9f50927)

As visible in the sub-region graph, “economic growth” appears most commonly in North American NDCs; from the map, we can observe that this is primarily accounted for by the United States. The phrase is least prevalent in Northern Europe. It is interesting that the three clusterings of Pacific island states have such different counts for the phrase: Polynesia has the second-lowest count and Micronesia the second-highest of all NDCs, with Melanesia in the middle.  On the map, we observe that Brazil, Saudi Arabia, some sub-saharan African states, and Australia include some degree of “economic growth” discourse, but there are no clearly discernible broad regional patterns. In general, however, it is relatively infrequent in comparison with other terms, as will be explored below.


#### "Development"
![development - map](https://github.com/user-attachments/assets/9b2f55d4-7d68-4327-a14e-1b44905be6aa)
![development - graph - subregion](https://github.com/user-attachments/assets/3701ee6c-b85f-4d17-9dcc-7e98852a538e)
![development - graph - income](https://github.com/user-attachments/assets/9f9a1fb3-b19d-44fb-a2f3-52006f04dde3)

“Development” yields a much clearer picture. It is least prevalent in Northern and Western Europe, Australia and New Zealand, then Northern America and Eastern Europe. We should take note of the significant difference between the United States and Canada’s NDCs and Mexico’s, which has a much higher prevalence for the word, as indicated on the map. We can thus observe that those regions commonly referred to as the “Global North”—Europe, the U.S. and Canada, and Australia and New Zealand—are least concerned with “development” in their emissions-reduction commitments.

By contrast, the term is most prevalent in Asia, in particular Central Asia, then South-eastern and Southern Asia, closely followed by Sub-Saharan Africa. Per the map, China’s normalized count is relatively high, alongside several Central Asian states: Turkmenistan, Uzbekistan, Afghanistan, Tajikistan, and Kyrgyzstan. Thailand, the Philippines, Indonesia, and especially Papua New Guinea, moving east into the Pacific, also have notably high counts. Russia and Brazil, though lower on the scale than China, appear to have similarly high frequencies for the term. In Latin America and Africa, the frequency is consistently moderate to high.

The Income Level graph reveals, indeed, that high income states are the least likely to mention development in their NDCs, a phenomenon reflected in the relative scarcity of the term in North America, Europe, and Australia. From low to upper-middle income, however, income level seems not to be strongly correlated to prevalence, and regional trends appear to dominate, with development discourse most prevalent in Asia and the Pacific, then in Latin America and Africa, and least of all in the “Global North.”

#### "Differentiated Responsibilities"
![responsibilities](https://github.com/user-attachments/assets/42361e67-ad20-48fa-ad37-d0fe86be1938)
![responsibilities - subregion](https://github.com/user-attachments/assets/9d0eadb2-951e-40fe-9b02-2ebd878dba8f)
![responsibilities - income](https://github.com/user-attachments/assets/cee0953d-040a-43bb-a6ca-0dbcd617e289)

Notably, states likely to incur the most obligations based on the principle of “common but differentiated responsibilities” make the least reference to the phrase—it appears nowhere in any Western or Eastern European, Northern American, or Australian NDC. It appears most frequently in Western Asia, Latin America and the Caribbean, with particularly high rates in Brazil. The highest and the lowest income states, interestingly, are the least likely to mention it in comparison with middle-income states.


### Theme 3: Innovation, the Private Sector, and the State

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

The “private” graph presents the striking prevalence of private-sector solutions coming out of the United States, which is individually responsible for the high prevalence of the term in North American NDC’s. It appears, by contrast, almost nowhere in Europe or the major Asian states, Russia, China, and India. It scarcely appears, likewise, in Australia or Canada. It has some prevalence in Central Asian and Middle Eastern states, some sub-Saharan African states, some South-Asian and Pacific states, and Brazil, but nowhere is its prevalence comparable to that of the United States.

Interestingly, in spite of the over-representation of the term in the high-GDP state of the US, the relative lack of prevalence in other high-GDP states (such as European states and China) reveal no correlation between income and prevalence of “private” sector ideas. Reliance on the private sector appears, that is, more a matter of cultural than purely economic traits.

#### "Regulation"
![regulation - map](https://github.com/user-attachments/assets/237bb907-3db6-4731-be64-adb786ef7b1e)
![regulation - graph - subregion](https://github.com/user-attachments/assets/29ad570a-7ba7-4553-942a-a399ba21af95)

This map indicates that regulation is prevalent almost exclusively in European NDCs. It appears almost nowhere else.

### Theme 4: Energy

We then tested a set of words associated with **energy** policy, including "fossil fuels," "energy transition," and "clean energy."

#### "Fossil Fuels"
![Fossil Fuels](https://github.com/user-attachments/assets/d5f2cf12-e6b4-40ba-8f5a-3776ba6a42cf)
![Fossil Fuels - Graph](https://github.com/user-attachments/assets/ced0a6e3-4b07-4f0b-b716-11f38c0c000d)

The phrase “fossil fuels” is most prevalent in China, Afghanistan, and Brazil. It appears almost not at all in Sub-Saharan Africa, all of the Pacific states between India and Australia, Russia, Mexico, and Western Latin American states. It has some prevalence, by contrast, in the United States and Europe. On the Income Level graph, we observe indeed that high income countries are most likely to discuss fossil fuels in their NDCs, and low and lower-middle income states are significantly less likely to do so. 

#### "Energy Transition"
![Energy Transition](https://github.com/user-attachments/assets/37d9e460-55aa-45db-a450-ca2d28f9f1b3)
![Energy Transition - Income Level Graph](https://github.com/user-attachments/assets/2a55b3af-4328-43a6-a859-dfec6528a8dd)
![Energy Transition - Sub-Region Graph](https://github.com/user-attachments/assets/849745ba-4c6e-4bd5-8ebb-e04ca6e422ea)

Brazil, the United Kingdom, and Cuba appear to have the highest prevalence of the phrase “energy transition” by far; by contrast, the phrase appears almost nowhere in Europe, Asia, the Middle East, Sub-Saharan Africa, or the rest of Latin America. It appears with some frequency in North America and Australia. The Income Level graph reveals that high and upper-middle income countries use this phrase on average more frequently than low and lower-middle income countries. When we examine by region, the average is surprising—Northern America has by far the highest averaged normalized count, despite the fact that none of its states have as high frequency levels as Brazil and the UK, because rates are moderately high across the region rather than split between singular highs and many lows, as is the case in Latin America (Brazil’s region) and Europe (the UK’s).

#### "Clean Energy"
![Clean Energy](https://github.com/user-attachments/assets/7701a25b-9de5-40c1-b563-9b7b54c8c049)
![Clean Energy - Sub-Region Graph](https://github.com/user-attachments/assets/65a0649c-2f42-46d5-982e-063b4dd09a61)

The phrase “clean energy” is another term most prevalent by far in the United States, as confirmed on the regional graph. The North American States are seconded by Australia, but by a wide margin. The phrase appears almost nowhere else.

#### "Renewable"
![renewable](https://github.com/user-attachments/assets/3be13147-b0a3-4f6b-8823-f18e65c6982a)
![renewable - graph - subregion](https://github.com/user-attachments/assets/78e39c1d-53ba-4bbd-bdac-23d21a4b74e8)
![renewable - graph - income](https://github.com/user-attachments/assets/a95460c1-872b-46b4-9013-d1535ab46621)

“Renewable” occurs more frequently than “clean” as a denotation for “energy” across the board. It is least prevalent in Europe, Northern America, Eastern Asia, and Australia and New Zealand, and most prevalent in Polynesia, Micronesia, Melanesia, North Africa, and Western Asia, with a notably high prevalence in Saudi Arabia. No correlation can be identified based on income.

### Theme 5: Mitigation v. Adaptation

We next considered the prevalence of ideas like "mitigation," or reducing climate impacts, and "adaptation" to the effects of climate change.

#### "Mitigation"
![mitigation - map](https://github.com/user-attachments/assets/f9ad1a8b-e387-4da7-921e-1369ee5a6c2e)
![mitigation - graph - subregion](https://github.com/user-attachments/assets/2029b12d-9cb8-4d76-8611-cd056034b31d)
![mitigation - graph - income](https://github.com/user-attachments/assets/39a16e0b-49df-4a0b-9041-f86b50f0626e)


A cursory look at the graph reveals that “mitigation,” or prevention of climate change, is far more prevalent in the Southern Hemisphere than the Northern. Europe and Northern America have the lowest frequencies for the term. It is most prevalent in Sub-Saharan and Northern Africa, followed by South-eastern Asia and Latin America and the Caribbean. It is scarcely mentioned by the US, Canada, European States, Russia, or China. The Income Level graph indicates indeed that high income states have the fewest references to “mitigation,” and low-income states, the most.

#### "Adaptation"
![Adaptation](https://github.com/user-attachments/assets/b9310e27-418a-4a53-baaa-9b9b69dbc52e)
![Adaptation - Sub-Region Graph](https://github.com/user-attachments/assets/8655cf6d-bc4d-41f0-b26c-335030eebfae)
![Adaptation - Income Level Graph](https://github.com/user-attachments/assets/681b7db4-5052-4a8f-8d22-3c83f4b6b522)

“Adaptation” reveals a slightly different picture. It is similarly the least prevalent in Europe, North America, and New Zealand, but Russia and China break from the “mitigation” trend and do make reference to “adaptation.” It is most prevalent in Melanesia, then Central Asia, followed by Northern Africa, Latin America and the Caribbean, South-eastern Asia, and finally Sub-Saharan Africa. Within Latin America, interestingly, Brazil includes rather few references while Argentina, Uruguay, Colombia, Ecuador, and Peru include more frequent notes.

### Theme 6: Land-Related Discourses

Finally, we look into terms that have more geographically specific relevance, such as "disaster," "agriculture," and "water."

#### "Disaster"
![Disaster](https://github.com/user-attachments/assets/d180ad76-39a0-4e8c-8746-033d364f6be6)
![Disaster - Sub Region Graph](https://github.com/user-attachments/assets/3134cda9-c48e-4739-9ab3-b9b04ffc0483)

A useful term to think about “adaptation” is “disaster,” as an index of references to climate impacts via natural disasters. Indeed, the results reveal an over-representation in the Pacific islands of Melanesia and Polynesia as well as South-eastern and Southern Asia. They appear scarcely at all in Europe—not at all in Western Europe—and minimally in Northern America or Australia and New Zealand. 

Although the regions with high “disaster” counts are notably exposed to natural disasters, infrequency of the word does not correlate to infrequency of hazards, as indicated by this map (source: https://journals.openedition.org/cybergeo/25297?lang=es): 

![Disaster Map](https://github.com/user-attachments/assets/45bec81a-065f-4316-8f22-567f5ed06d9b)

Moreover, high income states seem to have the lowest prevalence by far of the term, but the trend is not linear.

![Disaster - Income Level Graph](https://github.com/user-attachments/assets/3d5550f5-6a71-4937-9021-f11e11b1ca80)

#### "Agriculture"
![agriculture](https://github.com/user-attachments/assets/e0385802-90a2-487c-bda9-3a6bf945c926)
![agriculture - graph - subregion](https://github.com/user-attachments/assets/5ddaac88-de1c-4379-b343-4cc6e2439322)
![agriculture - graph - income](https://github.com/user-attachments/assets/35d695fd-48ed-4a2a-a728-95ed9a0f8342)

The word “agriculture” is most common in Central and Southern/Southeastern Asia and Sub-Saharan Africa. Observing the map, it appears to be a concern for many African states. By Income Level, we can observe that high income states are the least likely to discuss agriculture and low-income states most likely.

#### "Water"
![water](https://github.com/user-attachments/assets/6672bbe5-cb4f-4043-aa28-296a66fd6490)

Finally, we observe a strip of land in Central Asia, the Middle East, and Africa for whom “water” is a primary concern. Arid and water-scarce states such as Jordan, Iraq, and Saudi Arabia demonstrate particularly high frequency for the term, alongside Bolivia, where water politics are historically controvercial. Global North states reference water infrequently. The worldwide map does not exactly correlate to a map of water stress, although there is some overlap (source: World Resources Institute):

![23-08-02-aqueduct-4 0-launch-global-countries-baseline_Insights](https://github.com/user-attachments/assets/5b9aab85-bba7-41bb-9366-d5ea31e234ba)


## Regression Analysis
Following the counting words exercise, we were interested if there was any correlation between, for example, countries with higher GDP and their focus on "economic growth" in their emissions reductions strategies, or between SIDS and the mention of "disaster" in their emissions reductions strategies.

### Method
To determine any significant correlation between the explanatory variables 
--> insert code + painstaking process of having to complement to the csv file

### *insert* Chloe's beautiful heatmap

## Discussion

### Summary of Counting Words Findings

The first result of our counting words analysis is to demonstrate minimal integration of post- and degrowth discourses (Theme 1) into NDC commitments. Only three countries (Barbados, Cabo Verde, and Liberia) include the phrases “steady state” or “planetary boundaries,” and no countries mention “degrowth.” This insight may be useful to critics of economic growth in academic and civil society spaces, as it demonstrates a lack of conceptual penetration into national climate policy discussions.

Among terms from conventional economic discourse (Theme 2), we can observe that “economic growth” is overrepresented in North America, in particular in the United States, in contrast to Northern, Southern, and Eastern Europe’s relative disinterest in the concept. “Development” yields perhaps the most striking map: states in the “Global North” (Europe, the U.S., Canada, Australia and New Zealand) make the least mention of “development” in their emissions-reduction commitments by far. China, by contrast, mentions development frequently. One observes, in fact, strong development discourse in Asia, followed by Latin America and Africa. 

Another useful insight is that states of the Global North, who incur the most obligations under the principle, are least likely to make any reference to common but differentiated responsibilities—the term appears not at all in any Western or Eastern European, North American, or Australian NDC.

Theme 3 yields some targeted insights about country attitudes to innovation and the private sector. We observe the U.S.’s strong ideological commitment to innovation in comparison with the rest of the world. We also observe its prevalence in Saudi Arabia, China, Canada, and Brazil, and its relative absence in the rest of Asia, Europe, and Africa. High income states are significantly more likely to discuss “innovation” in their NDCs than low-income states. Alongside this insight, we can observe the strong prevalence of discussions of “technology” in Asian NDCs in comparison with the rest of the world. The U.S.’s strong prevalence of “innovation” is complemented by its frequent mentions of the “private” in comparison with almost every other state. References to the “private” do not reveal a correlation with income, as other high-income states (particularly in Europe) do not share the U.S.’s affinity for the term. We can observe, on the other hand, the strong prevalence of “regulation” in the European NDCs and its near absence in the rest of the world. 

Energy observations are more mixed; Brazil and China frequently mention “fossil fuels,” which appear relatively infrequently in low and lower middle income countries. “Energy transition” is a popular phrase in Brazil and the UK, while “clean energy” is more popular in the U.S. than anywhere else. Finally, “renewable,” which occurs overall more frequently than “clean” energy, is mentioned least frequently in Europe, North America, Australia and New Zealand, and Eastern Asia.

For Theme 5, we notice that mitigation, or climate change prevention, is far more prevalent in the Southern Hemisphere than the Northern. the US, Canada, Europe, Russia, and China scarcely mention it, while African, other Asian, and Latin American countries mention it relatively frequently. Low income countries reference mitigation in general far more than high-income countries. “Adaptation” is similarly absent in high-income state NDCs, but Russia and China break from their North American and European counterparts to mention it more frequently.

Finally, we can notice a high trend of discussing “disaster” in Asian and Pacific Island states. The mention-distribution does not however align with statistical disaster risk. High income states mention disaster least frequently. Agriculture is a significant concern for Central and South Asian and African states; again, more so for low-income than high-income states, while water appears in NDCs of states experiencing water scarcity in Africa, Central Asia, and the Middle East, alongside Bolivia, and scarcely at all in the Global North.

### Limitations

Our data is quite sparse -> tried to take into account the length of the NDC files.
For Europe, all EU countries have the same NDC -> which might modify the data, we decided it would be pertinent to replicate it, which might make it difficult (word choice in these NDCs might thus be exaggerated), we tried to factor this in by focusing on sub-regions as point of analysis.

## Conclusion



## References

Jernnäs, M. and Linnér, B. (2019) ‘A discursive cartography of nationally determined contributions to the Paris climate agreement’, Global Environmental Change, 55, pp. 73–83. Available at: https://doi.org/10.1016/j.gloenvcha.2019.01.006 

Savin, I. et al. (2025) ‘Analysing content of Paris climate pledges with computational linguistics’, Nature Sustainability, 8, pp. 297–306. Available at: https://doi.org/10.1038/s41893-024-01504-6 

Singh, P., Lehmann, E. and Tyrrell, M. (2024) ‘Climate Policy Transformer: Utilizing NLP to track the coherence of Climate Policy Documents in the Context of the Paris Agreement’, in D. Stammbach et al. (eds) Proceedings of the 1st Workshop on Natural Language Processing Meets Climate Change (ClimateNLP 2024). Bangkok, Thailand: Association for Computational Linguistics, pp. 1–11. Available at: https://doi.org/10.18653/v1/2024.climatenlp-1.1

Spokoyny, D. et al. (2024) ‘Aligning Unstructured Paris Agreement Climate Plans with Sustainable Development Goals’, in D. Stammbach et al. (eds) Proceedings of the 1st Workshop on Natural Language Processing Meets Climate Change (ClimateNLP 2024). Bangkok, Thailand: Association for Computational Linguistics, pp. 223–232. Available at: https://aclanthology.org/2024.climatenlp-1.17/ 
