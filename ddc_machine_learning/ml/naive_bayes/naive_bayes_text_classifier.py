class Text_Naive_Bayes():
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
        distinct = False,
        seperator = " ",
        cleaning_function = lambda x: x
    ):
        """
        Initialisation function for Naives Bayes on string data.
        
        ::param lower: (boolean) Flag to lower all words
        ::param distinct: (boolean) Flag if the words per text 
                                        are distinct or not
        ::param seperator: (string) String to seperate text to words
        ::param cleaning_function: (string) Function to clean text
        """
        self.cleaning_function = cleaning_function
        self.distinct = distinct
        self.lower = lower
        self.seperator = seperator
        self.word_counts_cat = {}
        self.word_counts = {}
        self.text_counts = {}
        self.clasification_scores = []


    def lower_distinct(self, text):
        """
        Function to clean a list of strings.
        Filters out empty strings.
        If lower = True:
            Makes every string lowercase.
        If distinct = True:
            Only looks at distinct words per text (no duplicates).
        
        ::param text: (dict[list])
        ::return: (list)
        """
        if self.distinct == True:
            text = list(set(text))
        if self.lower == True:
            text = [word.lower() for word in text]

        return [word for word in text if len(word) > 0]


    def get_words(self, data):
        """
        Function to get list of words from the text data.
        Also splits the text into their word components.

        ::param data: (dict[list])
        ::returns: (dict[list])
        ::returns: (list)
        """
        # Function to flatten lists
        word_function = lambda l: [item for sublist in l for item in sublist]
        words = []

        for key in data:
            data[key] = [self.cleaning_function(text).split(self.seperator) for text in data[key]]
 
            # Clean data
            text_data = []
            for text in data[key]:
                text_data += [self.lower_distinct(text)]
            
            data[key] = text_data
            
            # Flatten lists
            words += word_function(data[key])
        return data, words


    def get_words_unlabeled(self, data):
        """
        Function to get list of words from the text data (unlabeled).
        Also splits the texts into their word components.

        ::param data: (list[strings])
        ::returns: (dict[list])
        ::returns: (list)
        """
        data_ = []
        for text in data:
            text = self.cleaning_function(text).split(" ")
            text = self.lower_distinct(text)
            data_ += [text]

        return data_


    def get_text_counts(self, data):
        """
        Function to get number of strings per category.

        ::param data:
        ::returns: (dict)    
        """
        return {category:len(texts) for category, texts in data.items()}

    
    def naive_bayes_probability(self, text):
        """
        Function to get the score that a string is a 
        certain category.

        ::param text: (string)
        ::return: (dict[list])
        """
        score = ("temp", 0) # (category, score) - temporary
        scores = []
        
        for cat in self.word_counts_cat.keys():
            
            # Initial probabity (Percentage of strings that are this category)
            initial_prob = self.text_counts[cat]/sum(self.text_counts.values())
            
            # Only look at words that intersect with the fitted words
            word_inter = [val for val in text if val in self.word_counts.keys()]

            prob_multi = 1
            for word in word_inter:
                # Iteratively multiply each value by the percentage of words in a category
                prob_multi = prob_multi*(self.word_counts_cat[cat][word]/sum(self.word_counts_cat[cat].values()))#self.word_counts[word])
            prob = prob_multi*initial_prob
            
            scores += [(cat, prob)]
            
            # Update the score if its greater than initial
            if score[1] < prob:
                score = (cat, prob)
        return score[0], scores


    def naive_bayes(self, data):
        """
        Function to get the score that a text is a 
        certain category.

        ::param data: (list[string])
        ::return: (dict[list])
        """
        result = []
        scores = []
        
        for text in data:
            cat, score_temp = self.naive_bayes_probability(text)
            result += [cat]
            scores += [score_temp]
        
        # A list(tuples) with categories and scores
        self.clasification_scores = scores
        
        return result


    def word_counts_per_text(self, data, words):
        """
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

                # Adding 1 to all the word counts
                # This will prevent a case of 0 probability
                counts[word] = 1 

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
                # Adding 1 to all the word counts
                # This will prevent a case of 0 probability
                counts[word] = 1 
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


    def fit(self, data):
        """
        Fit Naive Bayes to the data.
        The fitted data should be of a dictionary the shape:
            category -> list of strings

        ::param data: (dict[list[list]]) dictionary of lists        
        """
        text = data.copy()
        
        # Get the words of the text
        text_data, words = self.get_words(text)

        # Create two dictionaries:
        #    words and correspondiong counts
        #    categories to words and corresponding counts
        word_counts, word_counts_cat = self.word_counts_per_text(text_data, words)
        
        # Get the number of strings per category
        text_counts = self.get_text_counts(text)

        self.word_counts = word_counts
        self.word_counts_cat = word_counts_cat
        self.text_counts = text_counts


    def classify(self, data):
        """
        Fit Naive Bayes to the data.

        ::param data: (list) list of text
        ::return: (list)
        """
        text = data.copy()
        
        # Get the words of the text
        words = self.get_words_unlabeled(text)

        return self.naive_bayes(words)


    def update(self, data):
        """
        Function to fit extra text  to the Naive Bayes classifer.
        Update the word_counts, word_counts_cat variables.

        ::param data: (dict[list[string]]) Dictionary of categories to lists of text
        """
        text = data.copy()
        
        # Split the data into words
        text, words = self.get_words(text)

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
                    self.word_counts_cat[cat][word] = 1
                    self.word_counts_cat[cat][word] += word_counts_cat_new[cat][word]

        # Update the total word counts
        for word in self.word_counts_cat[cat].keys():
            self.word_counts[word] = 0
            for cat in self.word_counts_cat.keys():
                self.word_counts[word] += self.word_counts_cat[cat][word]
        
        text_counts_new = self.get_text_counts(data)
        for cat in text_counts_new.keys():
            self.text_counts[cat] += text_counts_new[cat]