class Naive_Bayes():
    """
    Class for Naive Bayes operations of text data.

    ::param lower: (boolean) Flag to lower all words
    ::param distinct: (boolean) Flag if the words per text 
                                    are distinct or not
    ::param seperator: (string) String to seperate text to words
    ::param cleaning_function: (string) Function to clean text
    """

    def __init__(
        self,
        lower = True,
        distinct = True,
        seperator = " ",
        cleaning_function = lambda x: x,
        ignore_words = [],
        smoothing = 1
    ):
        """
        Initialisation function for Naives Bayes on string data.
        
        ::param lower: (boolean) Flag to lower all words
        ::param distinct: (boolean) Flag if the words per text 
                                        are distinct or not
        ::param seperator: (string) String to seperate text to words
        ::param cleaning_function: (string) Function to clean text
        ::param ignore_words: (string) Words that Naive Bayes should ignore
        ::param smoothing: (string) Number of words on default (Laplace Smoothing)
        """
        self.cleaning_function = cleaning_function
        self.distinct = distinct
        self.lower = lower
        self.seperator = seperator
        self.smoothing = smoothing
        self.ignore_words = ignore_words # Motivation from Carlos Amaral
        self.word_counts_cat = {}
        self.word_counts = {}
        self.text_counts = {}
        self.clasification_scores = []


    def lower_distinct(self, text):
        """
        # Data Cleaning
        Function to clean a list of strings.
        Filters out empty strings.
        If self.lower = True:
            Makes every string lowercase.
        If self.distinct = True:
            Only looks at distinct words per text (no duplicates in text).
        
        ::param text: (dict[list])
        ::return: (list)
        """
        if self.distinct == True:
            text = list(set(text))
        if self.lower == True:
            text = [word.lower() for word in text]

        return [word for word in text if len(word) > 0]


    def split_into_words(self, data):
        """
        # Data Cleaning
        Function to get list of words from the text data.
        Also splits the text into their word components.
        
        ::param data: (dict[list])
        *::returns: (dict[list]) data, where the text is split into word components
        ::returns: (list) list of words in data 
        """
        data = data.copy()
        # Function to flatten lists
        word_function = lambda l: [item for sublist in l for item in sublist]
        words = []

        # Assert if its labeled (training) data, or unlabeled (classifying)
        # dict (key, value pairing) is traiing data
        if type(data) == dict:
            for key in data:
                # Updates the text data into words
                data[key] = [self.cleaning_function(text).split(self.seperator) for text in data[key]]
                data[key] = \
                    [[word for word in sublist if word not in self.ignore_words] for sublist in data[key]]

                # Clean data
                text_data = []
                for text in data[key]:
                    text_data += [self.lower_distinct(text)]

                data[key] = text_data

                # Flatten lists
                words += word_function(data[key])
            return data, list(set(words))

        # list is data to be classified
        elif type(data) == list:
            data_ = []
            for text in data:
                text = self.cleaning_function(text).split(" ")
                text = self.lower_distinct(text)
                data_ += [text]

            return data_
        
        # If its not a list or a dict there is an error
        else:
            raise Exception("""
                Data types should be a dict (for training) or a list (for classifying.)""")


    def text_count(self, data):
        """
        # Getting counts
        Function to get number of texts per category.

        ::param data: (dict)
        ::returns: (dict)    
        """
        return {category:len(texts) for category, texts in data.items()}

    
    def word_counts_per_text(self, data, words):
        """
        # Getting counts
        Function to get the words per text.
        Also splits the text into their word components.

        ::param data: (dict[list])
        ::param words: (list)
        ::returns: ([dict[int]] Dictionary of words -> counts
        ::returns: (dict[dict[int]]) Dictionary of 
                                        categories -> words -> counts
        """
        word_counts_cat = {}
        word_counts = {}

        for cat in data.keys():
            counts = {}
            for word in words:

                # Adding 1 to all the word counts (Laplace smoothing)
                # This will prevent a case of 0 probability
                counts[word] = self.smoothing 

                for text in data[cat]:
                    counts[word] += text.count(word)

            word_counts_cat[cat] = counts

        # Get a dictionary of words and total values
        # Not necessarily needed, but makes the code easier
        for word in words:
            word_counts[word] = 0

            for cat in word_counts_cat.keys():
                word_counts[word] += word_counts_cat[cat][word]

        return word_counts, word_counts_cat
    
    
    def get_cat_word_dicts(self, data, words):
        """
        # Getting counts
        Function to get a dictionary of categories, words and counts.

        ::param data: (list[string])
        ::param words: (list)
        ::returns: (dict[int]) Dictionary of words -> counts
        ::returns: (dict[dict[int]]) Dictionary of 
                                        categories -> words -> counts
        """
        word_counts_cat = {}
        word_counts = {}

        for cat in data.keys():
            counts = {}
            for word in words:
                counts[word] = 0 
                for text in data[cat]:
                    counts[word] += text.count(word)

            word_counts_cat[cat] = counts

        # Get a dictionary of words and total values
        # Not necessarily needed, but makes the code easier
        for word in words:
            word_counts[word] = 0
            for cat in word_counts_cat.keys():
                word_counts[word] += word_counts_cat[cat][word]

        return word_counts, word_counts_cat

    
    def naive_bayes_probability(self, text):
        """
        # Classification
        Function to get the score that a string is a 
        certain category.

        ::param text: (string)
        ::return: (dict[list])
        """
        scores = []

        for cat in self.word_counts_cat.keys():
            # Initial probabity (Percentage of strings that are this category)
            # p_ -> probability, _w -> word, _c -> category
            initial_prob = self.text_counts[cat]/sum(self.text_counts.values())
            
            # Only look at words that intersect with the fitted words
            word_inter = [val for val in text if val in self.word_counts.keys()]

            prob_multi = 1
            prob_divide = 1
            p_w = {cat_:(self.text_counts[cat_]/(sum(self.text_counts.values()))) 
                        for cat_ in list(self.word_counts_cat.keys())}            
            for word in word_inter:
               
                for cat_ in self.word_counts_cat.keys():
                    p_w[cat_] = p_w[cat_]*(self.word_counts_cat[cat_][word]/(self.text_counts[cat_]+self.smoothing))

                p_w_c = self.word_counts_cat[cat][word]/(self.text_counts[cat]+self.smoothing)
                p_c =  initial_prob
                prob_multi = prob_multi*p_w_c
            prob_divide = sum(p_w.values())
            prob = prob_multi*initial_prob/prob_divide
            
            scores += [(cat, prob)]
            
        return scores


    def naive_bayes(
        self, data, category, weight):
        """
        # Classification
        Function to get the score that a text is a 
        certain category.
        Either returns the categories with the highest Naive Bayes scores,
            or returns a boolean list checking if a category is greater than a weight.

        ::param data: (list[string])
        ::return: (list[string OR boolean])
        """
        scores = []
        
        for text in data:
            score = self.naive_bayes_probability(text)
            scores += [score]
        
        # A list(tuples) with categories and scores
        self.clasification_scores = scores
        
        # Return the category by comparing the Naive Bayes scores against the other categories
        if category == None:
            return [sc[0] for sc in [max(l, key=lambda x:x[1]) for l in scores]]
        
        # Is the Naive Bayes greater/equal than the weight provided?
        else:
            category_score = [
                [score[1] for score in score_tuples if score[0] == category][0]
                    for score_tuples in scores]
            
            return [score >= weight for score in category_score]
            

    def fit(self, data):
        """
        # Classification
        Fit Naive Bayes to the data.
        The fitted data should be of a dictionary the shape:
            category -> list of strings

        ::param data: (dict[list[list]]) dictionary of lists        
        """
        text = data.copy()
        
        # Get the words of the text
        text_data, words = self.split_into_words(text)

        # Create two dictionaries:
        #    words and correspondiong counts
        #    categories to words and corresponding counts
        word_counts, word_counts_cat = self.word_counts_per_text(text_data, words)
        
        # Get the number of strings per category
        text_counts = self.text_count(text)

        self.word_counts = word_counts
        self.word_counts_cat = word_counts_cat
        self.text_counts = text_counts


    def classify(
        self, 
        data,
        category = None, 
        weight = 0.5
    ):
        """
        # Classification
        Fit Naive Bayes to the data.

        ::param data: (list) list of text
        ::return: (list)
        """
        text = data.copy()
        
        # Get the words of the text
        words = self.split_into_words(text)

        return self.naive_bayes(words, category, weight)


    def update(self, data):
        """
        # Update model
        Function to fit extra text  to the Naive Bayes classifer.
        Update the word_counts, word_counts_cat variables.

        ::param data: (dict[list[string]]) Dictionary of categories to lists of text
        """
        text = data.copy()
        
        # Split the data into words
        text, words = self.split_into_words(text)

        # Create two dictionaries:
        #    words and correspondiong counts
        #    categories to words and corresponding counts
        word_counts_new, word_counts_cat_new = \
            self.get_cat_word_dicts(text, words)

        # Update the Class dictionaries of words counts for categories
        for word in words:
            # Update the exisiting words in the dictionaries
            if word in self.word_counts.keys():
                for cat in self.word_counts_cat.keys():
                    self.word_counts_cat[cat][word] += word_counts_cat_new[cat][word]

            # Update the dictionaries with new words
            elif word not in self.word_counts.keys():
                for cat in self.word_counts_cat.keys():
                    self.word_counts_cat[cat][word] = self.smoothing
                    self.word_counts_cat[cat][word] += word_counts_cat_new[cat][word]

        # Update the total word counts
        for word in self.word_counts_cat[cat].keys():
            self.word_counts[word] = 0
            for cat in self.word_counts_cat.keys():
                self.word_counts[word] += self.word_counts_cat[cat][word]
        
        text_counts_new = self.text_count(data)
        for cat in text_counts_new.keys():
            self.text_counts[cat] += text_counts_new[cat]