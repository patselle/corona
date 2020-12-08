import os
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

class Patient():

    # This is the init method and is called every time an instance of this class is created
    def __init__(self):

        print('Erstelle Patient ...')

        # Declare list of symptoms
        self.symptoms_key_vec = [
            'Fieber',
            'Husten (produktiv)',
            'Husten (trocken)',
            'Atemnot',
            'Schnupfen',
            'Niesreiz',
            'Gliederschmerzen',
            'Abgeschlagenheit',
            'Halsschmerzen',
            'Kopfschmerz',
            'Augenjucken / -irritation']

        self.symptoms_val_vec = []

        # Enter symptoms via keyboard input and append it to list symptoms_val_vec
        print('Eingabe: \n\t 0: keine Angabe \n\t 1: fast nie \n\t 2: selten \n\t 3: möglich \n\t 4: häufig \n\t 5: fast immer')
        for symptom in self.symptoms_key_vec:
            while True:
                print('Habe Sie {}?'.format(symptom))
                try:
                    value = int(input())
                except:
                    print('Falsche Eingabe, bitte Zahl zwischen 0 und 5 eingeben ... ')
                    continue

                if value not in range(6):
                    print('Falsche Eingabe, bitte Zahl zwischen 0 und 5 eingeben ... ')

                else:
                    self.symptoms_val_vec.append(value)
                    break


    def print_symptoms(self):
        print('Übersicht aller Symptome')
        for key, value in zip(self.symptoms_key_vec, self.symptoms_val_vec):
            print('{}: {}'.format(key, value))


    def manhatten(self, vec1):

        distance = 0
        for entry1, entry2 in zip(vec1, self.symptoms_val_vec):
            if entry2 is not 0:
                distance -= (abs(entry1 - entry2))

        return distance

    def euclidian(self, vec1):

        distance = 0

        for entry1, entry2 in zip(vec1, self.symptoms_val_vec):
            if entry2 is not 0:
                distance += (entry1 - entry2)**2

        return - 1.0 * distance**(0.5)


    def cos_similarity(self, vec1):

        # Create two empty lists which represents our two vectors for which the distance/similarity will be computed
        tmp_vec1 = []
        tmp_vec2 = []

        # iterate over the sympomts of the disease as well as patients sympomts and exclude the 0's.
        # Do not consider the 0 in the patient symptoms list, since this corresponds to no specification
        for entry1, entry2 in zip(vec1, self.symptoms_val_vec):
            if entry2 is not 0:
                tmp_vec1.append(entry1)
                tmp_vec2.append(entry2)

        # This is the implementation of the cosinus similarity, see scipy
        return 1 - spatial.distance.cosine(tmp_vec1, tmp_vec2)


    def get_probabilities(self, x):
        # I use the softmax function to obtain probabilities
        e_x = np.exp(x) / np.sum(np.exp(x), axis=0)
        return (100 * e_x).astype(int)


    def allergy(self):
        name = 'allergy'
        # Each number corresponds to a symptom, the number itself to the specification
        # 1 never, 5 very often
        disease_symptoms = [1, 2, 4, 4, 5, 5, 1, 3, 2, 2, 5]
        return name, disease_symptoms


    def covid(self):
        name = 'covid-19'
        disease_symptoms = [4, 2, 4, 4, 2, 1, 2, 3, 3, 3, 1]
        return name, disease_symptoms


    def influenza(self):
        name = 'influenza'
        disease_symptoms = [4, 4, 5, 3, 2, 2, 5, 5, 3, 4, 1]
        return name, disease_symptoms


    def viral_infection(self):
        name = 'viral inf.'
        disease_symptoms = [2, 4, 2, 2, 5, 4, 4, 4, 3, 5, 4]
        return name, disease_symptoms


    def get_diagnose(self):

        # This list containts four objects for our four diseases
        # The implementation of these are the four methods see above and returns the name/symptoms respectively
        diseases = [
            self.allergy,
            self.covid,
            self.influenza,
            self.viral_infection]

        # Create an array of shape 4 and 3 (4 rows, 3 columns)
        # Each disease corresponds to a row and each column for one of the three implemented methods to measure the distance/similarity of the symptoms
        np_diagnoses = np.ones(shape=(len(diseases), 3))        
        
        # Create an empty list
        names = []

        # Iterate over all four diseases to compare their symptoms with the patients symptoms
        for i in range(len(diseases)):
            
            # Get name and symptoms of the current disease
            name, disease_symptoms = diseases[i]()

            # Append name to the list names
            names.append(name)

            # The first methode to evaluate the symptoms of the patient with them of the current disease is the manhatten or l1 norm
            # Write the result in the i-th row and the first column
            np_diagnoses[i, 0] = self.manhatten(disease_symptoms)

            # The second methode to evaluate the symptoms of the patient with them of the current disease is the euclidian distance or l2 norm
            # Write the result in the i-th row and the second column
            np_diagnoses[i, 1] = self.euclidian(disease_symptoms)

            # The second methode to evaluate the symptoms of the patient with them of the current disease is the so called cosinus similarity
            # Write the result in the i-th row and the third column
            np_diagnoses[i, 2] = self.cos_similarity(disease_symptoms)


        print(np_diagnoses[:, 0])
        # Scale the values of each column in the range of -1 and +1
        np_diagnoses[:,0] = np.interp(np_diagnoses[:,0], (np_diagnoses[:,0].min(), np_diagnoses[:,0].max()), (-1, +1))
        np_diagnoses[:,1] = np.interp(np_diagnoses[:,1], (np_diagnoses[:,1].min(), np_diagnoses[:,1].max()), (-1, +1))
        np_diagnoses[:,2] = np.interp(np_diagnoses[:,2], (np_diagnoses[:,2].min(), np_diagnoses[:,2].max()), (-1, +1))

        # Example: symptoms of the patient are the same as the viral inf. symptoms
        # The l1 norm unscaled looks: [-17. -22. -17.   0.]
        # 0 means both vectors are equal and -22 is the vektor with the worst agreement, covid in this case
        # When we scale this l1 vector it looks: [-0.54545455 -1.         -0.54545455  1.        ]
        # -17 -> -0.54545455
        # -22 -> -1.
        # -17 -> -0.54545455
        # 0 goes to 1.

        # But why we do so? See this example
        '''
        When we enter all symptoms for viral infection we get without scaling the following output

        Die Diagnose lautet: 	l1 Norm 	l2 Norm 	Cosinus Similarity
        ------------------------------------------------------------------------------
        allergy: 		        0.0 % 		0.0 % 		24.0 %
        covid-19: 		        0.0 % 		0.0 % 		23.0 %
        influenza: 		        0.0 % 		0.0 % 		24.0 %
        viral inf.: 		    99.0 % 		99.0 % 		27.0 %

        We expect a high outcome for viral inf but can we realy exclude the others with 100% guarantee, like l1 and l2?
        I don't think so!
        So, I think, l1 and l2 are to sensitive and the cosinus similarity is to non-sensitive (only 27%, but still the best)
        And this is the reason why I scale all columns, the values are closer to each other and gives the following outcome:

        Die Diagnose lautet: 	l1 Norm 	l2 Norm 	Cosinus Similarity
        ------------------------------------------------------------------------------
        allergy: 		        13.0 % 		12.0 % 		16.0 %
        covid-19: 		        8.0 % 		8.0 % 		8.0 %
        influenza: 		        13.0 % 		11.0 % 		15.0 %
        viral inf.: 		    64.0 % 		66.0 % 		60.0 %
        '''

        # Convert all values into probabilities 
        # To do so, create an array only containing ones of the same shape like np_diagnoses: (4,3)
        np_probs = np.ones_like(np_diagnoses)

        # For each column/disease call the get_probabilities() method
        np_probs[:,0] = self.get_probabilities(np_diagnoses[:,0])
        np_probs[:,1] = self.get_probabilities(np_diagnoses[:,1])
        np_probs[:,2] = self.get_probabilities(np_diagnoses[:,2])

        print('Die Diagnose lautet: \tl1 Norm \tl2 Norm \tCosinus Similarity')
        print('------------------------------------------------------------------------------')
        for i in range(np_probs.shape[0]):
            print('{}: \t\t{} % \t\t{} % \t\t{} %'.format(names[i], np_probs[i,0], np_probs[i,1], np_probs[i,2]))



if __name__ == "__main__":

    print('Hello from Corona detector')

    # Create an instance from the class Patient, call it pat
    pat = Patient()
    
    # Call the print_symptoms() method
    pat.print_symptoms()

    # Call the get_diagnose() method
    pat.get_diagnose()
